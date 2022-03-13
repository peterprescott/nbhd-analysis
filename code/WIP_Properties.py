# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from nbhd import data, geometry, utils


# -

def get_translator(df, first_column="first", second_column="second"):
    graph = nx.from_pandas_edgelist(df, first_column, second_column)
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    translator = {n: list(sorted(g.nodes))[0] for g in subgraphs for n in g.nodes}
    return translator


def sounds_institutional(local_type):

    words = [
        "Education",
        "Terminal",
        "Station",
        "Services",
        "Port",
        "Oil",
        "Hosp",
        "Heli",
        "Electric",
    ]
    return any([w.lower() in local_type.lower() for w in words])


db = data.Base()
pixel = utils.get_pixel("l13 7eq", db)


# +
def calculate_nonparametric_features(pixel: Polygon, db: data.Base):
    """Calculate non-parametric features.

    Parameters
    ----------
    pixel : Polygon
        pixel
    db : data.Base
        db
    """

    all_properties = db.knn("properties", "buildings", pixel)
    properties = all_properties.loc[all_properties.dist.eq(0)]
    properties = add_building_stats(properties)
    properties = find_stacked_properties(properties, pixel, db)
    properties = add_building_types(properties, pixel, db)

    return properties


def add_building_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    """add_building_stats.

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe

    Returns
    -------
    pd.DataFrame

    """

    properties_per_building = dict(dataframe.buildings_id.value_counts())
    dataframe["num_properties_in_building"] = dataframe.buildings_id.apply(
        lambda x: properties_per_building.get(x)
    )
    dataframe["building_footprint"] = gpd.GeoSeries.from_wkb(
        dataframe.buildings_geometry
    ).area
    dataframe["footprint_per_property"] = (
        dataframe.building_footprint / dataframe.num_properties_in_building
    )
    return dataframe


def find_stacked_properties(properties_df: pd.DataFrame, 
        pixel: Polygon, db: data.Base):
    """find_stacked_properties.

    Parameters
    ----------
    properties_df : pd.DataFrame
        properties_df
    pixel : Polygon
        pixel
    db : data.Base
        db
    """

    nn_properties = db.knn("properties", "properties", pixel)
    stacked_properties = nn_properties.loc[nn_properties.dist.eq(0)]
    stacked_properties.columns = [
        "first",
        "first_geom",
        "second",
        "second_geom",
        "dist",
    ]
    translator = get_translator(stacked_properties)
    stacked_properties["stacked"] = stacked_properties["first"].apply(
        lambda x: translator.get(x)
    )
    stacked_dict = dict(zip(stacked_properties["first"], stacked_properties.stacked))
    stacked_counts = dict(stacked_properties.stacked.value_counts())
    pid_stacked_counts = dict(
        zip(
            stacked_properties["first"],
            stacked_properties.stacked.apply(lambda x: stacked_counts.get(x)),
        )
    )
    properties_df["stacked"] = properties_df.properties_id.apply(
        lambda x: stacked_dict.get(x)
    )
    properties_df["stacked_count"] = properties_df.properties_id.apply(
        lambda x: pid_stacked_counts.get(x, 0)
    )

    return properties_df


def add_building_types(dataframe: pd.DataFrame, pixel: Polygon, 
        db: data.Base):
    """add_building_types.

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe
    pixel : Polygon
        pixel
    db : data.Base
        db
    """

    nn_bn = db.knn("buildings", "names", pixel, t2_columns=["local_type"])
    nn_bn["institution"] = nn_bn.local_type.apply(lambda x: sounds_institutional(x))
    institutional_buildings = dict(zip(nn_bn.buildings_id, nn_bn.institution))
    institution_type = dict(zip(nn_bn.buildings_id, nn_bn.local_type))
    dataframe["institution"] = dataframe.buildings_id.apply(
        lambda x: institutional_buildings.get(x) * institution_type.get(x)
    )
    dataframe.institution = dataframe.institution.apply(
        lambda x: None if x == "" else x
    )
    return dataframe


# -

df = calculate_nonparametric_features(pixel, db)

df.loc[~df.buildings_id.duplicated()].institution.value_counts()


# +
# for given faceblock
# find number of neighbouring faceblocks
# find number of properties on neighbouring faceblocks

def count_neighbours(row, df):
    
    df = df.loc[~df.roads_id.duplicated()].copy()
    start, end = row.startNode, row.endNode
    df = df.loc[(df.startNode==start) | (df.startNode==end) | (df.endNode==start) | (df.endNode==end)]
    df = df.loc[df.roads_id!=row.roads_id]
    neighbouring_faceblocks = len(df)
    properties_on_neighbouring_faceblocks = df.properties_on_road.sum()
    
    return pd.Series({'roads_id':row.roads_id, 
                      'neighbouring_faceblocks': neighbouring_faceblocks,
           'properties_on_neighbouring_faceblocks': properties_on_neighbouring_faceblocks})


# +
roads = db.intersects("roads", pixel)

nn_pr = db.knn(
    "properties",
    "roads",
    t2_columns=['"startNode"', '"endNode"', "name1", "length", 'road_function'],
    polygon=pixel,
)
property_counts_dict = dict(nn_pr.value_counts('roads_id'))
nn_pr['properties_on_road'] = nn_pr.roads_id.apply(lambda x: property_counts_dict.get(x,0))
nn_pr['length_per_property'] = nn_pr.length / nn_pr.properties_on_road
nn_pr['log_length_per_property'] = np.log(nn_pr.length_per_property)
neighbours = nn_pr.loc[~nn_pr.roads_id.duplicated()].apply(axis=1, func=lambda row: count_neighbours(row,nn_pr))
# -

nn_pr = nn_pr.merge(neighbours, on='roads_id')

roads = roads.rename(columns={'id':'roads_id'})
roads = roads[[c for c in nn_pr.columns if c in roads.columns]]

roads






