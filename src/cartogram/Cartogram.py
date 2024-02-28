import colorsys
import os
from functools import cache, cached_property

import geopandas as gpd
import matplotlib.pyplot as plt
from gig import Ent, EntType
from shapely.geometry import MultiPolygon, Polygon
from utils import Log

log = Log('Cartogram')


def custom_timer(func):
    def wrapper(*args, **kwargs):
        import time

        t_start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        dt_ns = time.perf_counter_ns() - t_start
        dt = dt_ns / 1e9
        if dt > 1:
            log.debug(f'{func.__name__}: {dt:.2f}s')
        return result

    return wrapper


class Cartogram:
    # Construction

    def __init__(self, id_to_exp_area: dict[str, float] = {}):
        self.id_to_exp_area = id_to_exp_area
        self.id_to_multipolygon = self.get_init_id_to_multipolygon()

    @cached_property
    def ent_ids(self):
        return list(self.id_to_exp_area.keys())

    @cached_property
    def ents(self):
        return [Ent.from_id(id) for id in self.ent_ids]

    # Init Shape Data
    @staticmethod
    def lnglat_key(lnglat):
        return f'{lnglat[0]:.4f},{lnglat[1]:.4f}'

    @staticmethod
    def raw_geo_to_multipolygon(raw_geo):
        max_raw_polygon = max(
            raw_geo, key=lambda raw_polygon: len(raw_polygon)
        )
        return MultiPolygon([Polygon(max_raw_polygon)])

    def get_init_id_to_multipolygon(self):
        return {
            ent.id: Cartogram.raw_geo_to_multipolygon(ent.get_raw_geo())
            for ent in self.ents
        }

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
    def do_scale_multipolygon(idx, multipolygon, centroid, scale_factor):
        lng0, lat0 = centroid
        for polygon in multipolygon.geoms:
            for lng, lat in polygon.exterior.coords:
                k = Cartogram.lnglat_key((lng, lat))
                if k not in idx:
                    idx[k] = [0, 0]
                dlng, dlat = lng - lng0, lat - lat0
                idx[k][0] += (lng0 + dlng * scale_factor) - lng
                idx[k][1] += (lat0 + dlat * scale_factor) - lat

        return idx

    @custom_timer
    def remap(self, idx):
        for id in self.ent_ids:
            multipolygon = self.id_to_multipolygon[id]

            new_polygon_list = []
            for polygon in multipolygon.geoms:
                new_lnglat_list = []
                for lng, lat in polygon.exterior.coords:
                    k = Cartogram.lnglat_key((lng, lat))
                    lng1, lat1 = lng, lat
                    if k in idx:
                        dlng, dlat = idx[k]
                        lng1 += dlng
                        lat1 += dlat
                    new_lnglat_list.append((lng1, lat1))
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

    # Scale
    @custom_timer
    def scale(self):
        LEARNING_RATE = 0.2
        N_GENS = 10
        MIN_MAE = 0.01
        for g in range(N_GENS):
            c.write(os.path.join('images', f'cartogram-scaled-{g}.png'))
            log.info(f'{g}) MAE={self.mae:.3f}')
            if self.mae < MIN_MAE:
                break
            idx = {}
            for id, exp_scale_factor in self.id_to_exp_scale_factor.items():
                scale_factor = 1 + (exp_scale_factor - 1) * LEARNING_RATE
                centroid = self.id_to_centroid[id]
                idx = Cartogram.do_scale_multipolygon(
                    idx, self.id_to_multipolygon[id], centroid, scale_factor
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
    @cache
    def get_color(id):
        h = id.__hash__()
        h = (h % 240) / 360.0
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
            geo.plot(ax=ax, color=color)
            ax.text(*self.id_to_centroid[id], id, fontsize=12)
        plt.savefig(file_path)
        log.debug(f'Wrote {file_path}')


if __name__ == '__main__':
    ent_ids = Ent.ids_from_type(EntType.DISTRICT)
    ent_ids = [id for id in ent_ids if ('LK-1' in id or 'LK-9' in id)]

    print(ent_ids)
    c = Cartogram({id: 1 for id in ent_ids})

    c.scale()
