# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from nbhd import data, geometry, utils

# %%
db = data.Base()

# %%
pixel = utils.get_pixel('l13 7eq', db)

# %%
all_properties = db.knn('properties', 'buildings', pixel)

# %%
properties = all_properties.loc[all_properties.dist.eq(0)]

# %%
nn_properties = db.knn('properties', 'properties', pixel)

# %%
stacked_properties = nn_properties.loc[nn_properties.dist.eq(0)]

# %%
import networkx as nx


# %%
def get_translator(df, first_column='first', second_column='second'):
    graph = nx.from_pandas_edgelist(df, first_column, second_column)
    subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    translator = {n: list(sorted(g.nodes))[0] for g in subgraphs for n in g.nodes}
    return translator


# %%
stacked_properties.columns = ['first', 'first_geom', 'second', 'second_geom', 'dist']

# %%
translator = get_translator(stacked_properties)

# %%
stacked_properties['stacked'] = stacked_properties['first'].apply(lambda x: translator.get(x))

# %%
stacked_dict = dict(zip(stacked_properties['first'], stacked_properties.stacked))

# %%
stacked_counts = dict(stacked_properties.stacked.value_counts())

# %%
pid_stacked_counts = dict(zip(stacked_properties['first'],stacked_properties.stacked.apply(lambda x: stacked_counts.get(x))))

# %%
properties['stacked'] = properties.properties_id.apply(lambda x: stacked_dict.get(x))

# %%
properties['stacked_count'] = properties.properties_id.apply(lambda x: pid_stacked_counts.get(x,0))

# %%
properties_per_building = dict(properties.buildings_id.value_counts())

# %%
properties['num_properties_in_building'] = properties.buildings_id.apply(
    lambda x: properties_per_building.get(x))

# %%
import geopandas as gpd

# %%
properties['building_footprint'] = gpd.GeoSeries.from_wkb(properties.buildings_geometry).area

# %%
properties['footprint_per_property'] = properties.building_footprint / properties.num_properties_in_building

# %%
properties.columns

# %%
nn_bn = db.knn('buildings','names', pixel, t2_columns=['local_type'])


# %%
def sounds_institutional(local_type):
    
    words = ['Education', 'Terminal', 'Station', 'Services', 'Port', 'Oil',
            'Hosp', 'Heli', 'Electric']
    return any([w.lower() in local_type.lower() for w in words])


# %%
nn_bn['institution'] = nn_bn.local_type.apply(lambda x: sounds_institutional(x))
institutional_buildings = dict(zip(nn_bn.buildings_id, nn_bn.institution))

# %%
institution_type = dict(zip(nn_bn.buildings_id, nn_bn.local_type))

# %%
properties['institution'] = properties.buildings_id.apply(
    lambda x: institutional_buildings.get(x) * institution_type.get(x))

# %%
properties.institution = properties.institution.apply(lambda x: None if x == '' else x)

# %%
roads = db.intersects('roads',pixel)

# %%
roads.columns

# %%
nn_pr = db.knn('properties','roads',t2_columns=['"startNode"','"endNode"','name1'],polygon=pixel)

# %%
nn_pr.value_counts('roads_id')

# %%
nn_pr.name1
