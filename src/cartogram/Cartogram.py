import colorsys
import os
from functools import cache, cached_property

import geopandas as gpd
import matplotlib.pyplot as plt
from gig import Ent, EntType
from shapely.geometry import MultiPolygon, Polygon
from utils import File, Log

import topojson as tp

log = Log('Cartogram')
os.makedirs('images', exist_ok=True)
os.makedirs('topojson', exist_ok=True)


def custom_timer(func):
    def wrapper(*args, **kwargs):
        import time

        t_start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        dt_ns = time.perf_counter_ns() - t_start
        dt_ms = round(dt_ns / 1e6, 0)
        if dt_ms > 100:
            log.debug(f'⌚ {func.__name__} - {dt_ms:,}ms')
        return result

    return wrapper


class Cartogram:
    LEARNING_RATE = 0.1
    N_GENS = 10
    MIN_MAX_ERROR = 0.01
    # Construction

    def __init__(self, id_to_exp_area: dict[str, float] = {}):
        self.id_to_exp_area = id_to_exp_area
        self.id_to_multipolygon = self.get_init_id_to_multipolygon()

    @staticmethod
    def from_ent_ids(ent_ids):
        ents = [Ent.from_id(id) for id in ent_ids]
        id_to_exp_area = {ent.id: ent.population for ent in ents}
        return Cartogram(id_to_exp_area)

    @cached_property
    def ent_ids(self):
        return list(self.id_to_exp_area.keys())

    @cached_property
    def ents(self):
        return [Ent.from_id(id) for id in self.ent_ids]

    # Init Shape Data
    @staticmethod
    def lnglat_key(lnglat):
        return ','.join(map(lambda x: f'{x:.8f}', lnglat))

    @staticmethod
    def lnglat_unkey(k):
        return tuple(map(float, k.split(',')))

    @staticmethod
    def lnglat_normalize(lnglat):
        return Cartogram.lnglat_unkey(Cartogram.lnglat_key(lnglat))

    @staticmethod
    def raw_geo_to_multipolygon(raw_geo):
        max_raw_polygon = max(
            raw_geo, key=lambda raw_polygon: len(raw_polygon)
        )
        max_raw_polygon = [
            Cartogram.lnglat_normalize(lnglat) for lnglat in max_raw_polygon
        ]
        max_polygon = Polygon(max_raw_polygon)

        return MultiPolygon([max_polygon])

    @staticmethod
    @custom_timer
    def fix_HACK(ent_ids, id_to_multipolygon):
        multipolygons = list(id_to_multipolygon.values())
        geo = gpd.GeoDataFrame(geometry=gpd.GeoSeries(multipolygons))
        topo = tp.Topology(geo, prequantize=False).to_json()
        topo_path = os.path.join('topojson', 'all.json')
        File(topo_path).write(topo)
        log.debug(f'Wrote {topo_path}')
        geo2 = gpd.read_file(topo)
        multipolygons2 = geo2.geometry
        id_to_multipolygon = {
            id: multipolygon
            for id, multipolygon in zip(ent_ids, multipolygons2)
        }
        return id_to_multipolygon

    def get_init_id_to_multipolygon(self):
        id_to_multipolygon = {}
        for ent in self.ents:
            raw_geo = ent.get_raw_geo()
            multipolygon = Cartogram.raw_geo_to_multipolygon(raw_geo)
            id_to_multipolygon[ent.id] = multipolygon

        id_to_multipolygon = Cartogram.fix_HACK(
            self.ent_ids, id_to_multipolygon
        )

        return id_to_multipolygon

    # Scale Helpers
    @property
    @custom_timer
    def id_to_centroid(self):
        return {
            id: multipolygon.centroid.coords[0]
            for id, multipolygon in self.id_to_multipolygon.items()
        }

    @property
    def id_to_area(self):
        return {
            item[0]: item[1].area for item in self.id_to_multipolygon.items()
        }

    @property
    def id_to_exp_area_norm(self):
        total_exp_area = sum(self.id_to_exp_area.values())
        return {
            id: exp_area / total_exp_area
            for id, exp_area in self.id_to_exp_area.items()
        }

    @property
    def id_to_area_norm(self):
        total_area = sum(self.id_to_area.values())
        return {
            id: scaled_area / total_area
            for id, scaled_area in self.id_to_area.items()
        }

    @property
    def id_to_exp_scale_factor(self):
        return {
            id: self.id_to_exp_area_norm[id] / self.id_to_area_norm[id]
            for id in self.ent_ids
        }

    @staticmethod
    @custom_timer
    def do_scale_multipolygon(idx, scale_factor, multipolygon):
        centroid = multipolygon.centroid.coords[0]
        lng0, lat0 = centroid
        k_set = set()
        for polygon in multipolygon.geoms:
            for lng, lat in polygon.exterior.coords:
                k = Cartogram.lnglat_key((lng, lat))
                if k not in idx:
                    idx[k] = [0, 0]
                if k in k_set:
                    continue
                k_set.add(k)
                dlng, dlat = lng - lng0, lat - lat0
                lng1, lat1 = (
                    lng0 + dlng * scale_factor,
                    lat0 + dlat * scale_factor,
                )

                idx[k][0] += lng1 - lng
                idx[k][1] += lat1 - lat

        return idx

    @custom_timer
    def remap(self, idx):
        for id in self.ent_ids:
            multipolygon = self.id_to_multipolygon[id]
            new_polygon_list = []
            k_set = set()
            assert len(multipolygon.geoms) == 1
            for polygon in multipolygon.geoms:
                new_lnglat_list = []
                for lng, lat in polygon.exterior.coords:
                    lng1, lat1 = lng, lat
                    k = Cartogram.lnglat_key((lng, lat))
                    assert k in idx
                    dlng, dlat = idx[k]
                    lng1 += dlng * Cartogram.LEARNING_RATE
                    lat1 += dlat * Cartogram.LEARNING_RATE
                    k_set.add(k)

                    new_lnglat_list.append(
                        Cartogram.lnglat_normalize((lng1, lat1))
                    )
                new_polygon = Polygon(new_lnglat_list)
                new_polygon_list.append(new_polygon)

            self.id_to_multipolygon[id] = MultiPolygon(new_polygon_list)

    @property
    def mae(self):
        return sum(
            [
                abs(self.id_to_exp_area_norm[id] - self.id_to_area_norm[id])
                for id in self.ent_ids
            ]
        ) / len(self.ent_ids)

    @property
    def max_error(self):
        return max(
            [
                abs(self.id_to_exp_area_norm[id] - self.id_to_area_norm[id])
                for id in self.ent_ids
            ]
        )

    # Scale
    @custom_timer
    def scale(self):
        for g in range(Cartogram.N_GENS):
            c.write(os.path.join('images', f'cartogram-scaled-{g}.png'))
            log.info(f'{g}) max_error={self.max_error:.3f}')
            if self.max_error < Cartogram.MIN_MAX_ERROR:
                break
            idx = {}
            for id in self.ent_ids:
                exp_scale_factor = self.id_to_exp_scale_factor[id]
                log.debug(f'{id=}, {exp_scale_factor=}')

                idx = Cartogram.do_scale_multipolygon(
                    idx,
                    exp_scale_factor,
                    self.id_to_multipolygon[id],
                )
            self.remap(idx)

        c.write(os.path.join('images', 'cartogram-scaled-final.png'))

    # Write
    @staticmethod
    def multipolygon_to_geo(multipolygon):
        return gpd.GeoDataFrame(
            index=[0], crs='epsg:4326', geometry=[multipolygon]
        )

    @staticmethod
    def geo_to_topojson(
        geo,
    ):
        return tp.Topology(geo, prequantize=False).to_json()

    @staticmethod
    @cache
    def get_color(id):
        h = id.__hash__()
        h = (h % 20) / 20
        s = 1
        v = 0.75
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)

    def write(self, file_path: str):
        plt.close()
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        for id, multipolygon in self.id_to_multipolygon.items():
            geo = Cartogram.multipolygon_to_geo(multipolygon)

            color = Cartogram.get_color(id)
            geo.plot(ax=ax, facecolor=color, edgecolor='black')
            # ax.text(*self.id_to_centroid[id], id, fontsize=12)
        plt.savefig(file_path)
        log.debug(f'Wrote {file_path}')


if __name__ == '__main__':
    ent_type = EntType.PROVINCE
    parent_id = 'LK'

    ent_ids = [id for id in Ent.ids_from_type(ent_type)]

    print(ent_ids)
    c = Cartogram.from_ent_ids(ent_ids)

    print('id_to_exp_scale_factor', c.id_to_exp_scale_factor)

    c.scale()
