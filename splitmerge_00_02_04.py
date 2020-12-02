# ex OVERSPLIT

import numpy as np
import pandas as pd
from math import sqrt
import itertools
from sklearn.cluster import KMeans


# DISTANCE FUNCTIONS
squared_distance_list = lambda some_cloud, some_point: pd.DataFrame((some_cloud - some_point.values.squeeze()) ** 2).sum(axis=1)
squared_full_graph_list = \
    lambda some_cloud: list(itertools.chain.from_iterable(list(map(lambda every_row: squared_distance_list(some_cloud, some_cloud.iloc[every_row]), list(range(len(some_cloud)))))))

distance = lambda some_point_1, some_point_2: sqrt(pd.DataFrame((some_point_1.values.squeeze() - some_point_2.values.squeeze()) ** 2).values.sum())
def min_distance(some_cloud, some_point):
    distances = pd.DataFrame((some_cloud - some_point.values.squeeze()) ** 2).sum(axis=1).values
    distances = pd.DataFrame({'closest': some_cloud.index, 'min_distance': distances})
    distances = distances.loc[distances.loc[:, 'min_distance'] > 0, :]
    distances = distances.loc[distances.loc[:, 'min_distance'] == distances.loc[:, 'min_distance'].min()]
    distances.loc[:, 'min_distance'] = sqrt(distances.loc[:, 'min_distance'])
    return distances
min_distances = lambda some_cloud: pd.concat([min_distance(some_cloud, some_cloud.iloc[[i]]) for i in range(some_cloud.shape[0])])
get_attribute_value = lambda some_dataframe, some_ids, field_name, id_column_name='id': list(map(lambda every_id: some_dataframe.loc[some_dataframe.loc[:, id_column_name] == every_id, field_name].values[0], some_ids))


# SOME SPECIAL FUNCTIONS
def log_label_exchange(some_df, id_old, id_new, what_to_change = 'merged'):
    some_df.loc[some_df.loc[:, what_to_change] == id_old, what_to_change] = id_new
    return some_df

def label_update(some_centroid_distances, id_old, id_new):
    some_centroid_distances = log_label_exchange(some_centroid_distances, id_old, id_new, 'id')
    some_centroid_distances = log_label_exchange(some_centroid_distances, id_old, id_new, 'closest')
    return some_centroid_distances

count_clusters = lambda some_centroid_distances: some_centroid_distances.loc[:, 'id'].unique().shape[0]
min_cluster_volume = lambda some_centroid_distances: some_centroid_distances.groupby('id')['volume'].sum().min()

smaller_steps = lambda some_steps, pattern_step: some_steps.loc[some_steps.loc[:, 'step'] <= pattern_step, :]
#larger_steps = lambda some_steps, pattern_step: some_steps.loc[some_steps.loc[:, 'step'] >= pattern_step, :]

def get_relative_steps(some_steps, pattern_step):
    i = 0
    while some_steps.loc[i, 'step'] < pattern_step:
        i += 1
    return some_steps.loc[i:, :]


# SPLITMERGE CLASS DEFINITION
class splitmerge:
    def get_volumes(self, some_labels=None, label_column='source'):
        if some_labels is None:
            some_labels = self.source_labels
        volumes = some_labels[label_column].value_counts()
        volumes = pd.DataFrame(volumes)
        volumes.columns = ['volume']
        volumes.loc[:, 'fraction'] = volumes.loc[:, 'volume'] / some_labels.shape[0]
        volumes.loc[:, 'cluster'] = volumes.index
        return volumes

    def get_centroid_distances(self):
        self.centroids.index = [i for i in range(self.centroids.shape[0])]
        self.centroids.loc[:, 'id'] = self.centroids.index
        self.centroids.loc[:, 'volume'] = get_attribute_value(self.volumes, self.centroids.loc[:, 'id'], 'volume', 'cluster')
        all_relations = []
        for i in range(self.centroids.shape[0]):
            for j in range(i + 1, self.centroids.shape[0]):
                row_i = self.centroids.iloc[[i]]
                row_j = self.centroids.iloc[[j]]
                some_distance = distance(row_i, row_j)
                all_relations.append({
                    'id': row_i['id'].values[0],
                    'closest': row_j['id'].values[0],
                    'distance': some_distance,
                    'volume': row_i['volume'].values[0],
                    'is_closest': False
                })
                all_relations.append({
                    'id': row_j['id'].values[0],
                    'closest': row_i['id'].values[0],
                    'distance': some_distance,
                    'volume': row_j['volume'].values[0],
                    'is_closest': False
                })
        all_relations = pd.DataFrame(all_relations)
        all_relations = all_relations.loc[all_relations.loc[:, 'distance'] > 0, :]
        shortest_distances = all_relations.groupby('id')['distance'].min()
#        shortest_distances = pd.DataFrame({'distance': shortest_distances, 'radius': shortest_distances / 2})
        shortest_distances = pd.DataFrame({'distance': shortest_distances})
        if self.radii is None:
            shortest_distances.loc[:, 'radius'] = shortest_distances / 2
        else:
            shortest_distances = shortest_distances.join(self.radii)
        all_relations = all_relations.loc[all_relations.loc[:, 'distance'] <= shortest_distances['distance'].max(), :]
        all_relations.index = all_relations.loc[:, 'id']
        all_relations = all_relations.join(shortest_distances, rsuffix='_shortest')
        all_relations.loc[all_relations.loc[:, 'distance'] == all_relations.loc[:, 'distance_shortest'], 'is_closest'] = True
        all_relations.drop(['distance_shortest'], axis=1, inplace=True)

        all_relations.loc[:, 'neighbor_radius'] = get_attribute_value(all_relations,
                                                                        all_relations.loc[:, 'closest'],
                                                                        'radius', 'id')
        all_relations.loc[:, 'lambda'] = all_relations.loc[:, 'distance'] -\
                                                all_relations.loc[:, 'radius'] -\
                                                all_relations.loc[:, 'neighbor_radius']

        all_relations.index = [i for i in range(all_relations.shape[0])]
        return all_relations

    def cluster_merge(self, order_by='lambda'):
        label_map = pd.DataFrame([{'source': i, 'merged': i} for i in range(self.centroids.shape[0])])
        label_map.index = label_map.loc[:, 'source']
        merged_clusters_distances = self.centroid_distances

        collapse_analysis = []
        collapsed_labels = dict()
        merged_clusters_distances = merged_clusters_distances.sort_values(order_by, ascending=True)
        for i in merged_clusters_distances.index:
            id_i = merged_clusters_distances.loc[i, 'id']
            closest_i = merged_clusters_distances.loc[i, 'closest']
            if id_i != closest_i:
                new_id = min([id_i, closest_i])
                old_id = max([id_i, closest_i])
                merged_clusters_distances = label_update(merged_clusters_distances, old_id, new_id)
                label_map = log_label_exchange(label_map, old_id, new_id, 'merged')
                collapse_analysis.append({
                                             order_by: merged_clusters_distances.loc[i, order_by],
                                             'n_clusters': count_clusters(merged_clusters_distances),
                                             'min_cluster_volume': min_cluster_volume(merged_clusters_distances),
                                             'merged_cluster_volume': merged_clusters_distances.loc[i, 'volume'],
                                             'merged_cluster_id': id_i
                                           , 'merged_cluster_new_id': new_id
                                         })
                collapsed_labels[collapse_analysis[-1]['n_clusters']] = label_map.copy()
        return pd.DataFrame(collapse_analysis), collapsed_labels  # , merged_clusters_distances, label_map

    def measure_steps(self, key='lambda'):
        collapse_analysis = self.stepwise_merge
        steps = pd.DataFrame()

        # i = 0
        previous_item = 0
        for every_item in collapse_analysis.index:
            steps = pd.concat([steps, pd.DataFrame(
                {
                    'step': [collapse_analysis.iloc[every_item][key] - collapse_analysis.iloc[previous_item][key]],
                    'n_clusters': [int(collapse_analysis.iloc[every_item]['n_clusters'])]
                    , 'merged_volume': [int(collapse_analysis.iloc[every_item]['merged_cluster_volume'])]
                    , 'merged_id': [int(collapse_analysis.iloc[every_item]['merged_cluster_id'])]
                    , 'merged_new_id': [int(collapse_analysis.iloc[every_item]['merged_cluster_new_id'])]
                    # , 'first_larger': i
                }
            )])
            previous_item = every_item
            # i += 1

        steps.index = range(steps.shape[0])
        steps.loc[:, 'k_before_step'] = steps.loc[:, 'n_clusters'] + 1

        # absolute_risks = []
        is_larger_than_all_previous = []
        max_step = 0
        local_max_steps = []
        compare_critical_steps = []
        for i in range(steps.shape[0]):
            # current_step = steps.loc[i, 'step']
            # absolute_risks.append(float(smaller_steps(steps, current_step).shape[0]) / steps.shape[0])
            if steps.loc[i, 'step'] >= max_step:
                is_larger_than_all_previous.append(1)
                if max_step > 0:
                    compare_critical_steps.append(steps.loc[i, 'step'] / max_step)
                else:
                    compare_critical_steps.append(0)
                max_step = steps.loc[i, 'step']
            else:
                is_larger_than_all_previous.append(0)
                compare_critical_steps.append(0)
            local_max_steps.append(max_step)

        # steps.loc[:, 'absolute_risk'] = absolute_risks
        steps.loc[:, 'local_max_step'] = local_max_steps
        steps.loc[:, 'most_risky'] = is_larger_than_all_previous
        steps.loc[:, 'compare_critical_steps'] = compare_critical_steps

        steps = steps.sort_values(['step'], ascending=False)
        steps.loc[:, 'almost_max'] = steps.loc[:, 'step'] / steps.loc[:, 'local_max_step']
        steps.loc[:, 'most_risky_rank'] = steps.loc[:, 'most_risky'] * steps.loc[:, 'most_risky'].cumsum()
        # steps = steps.sort_values('n_clusters', ascending=False)
        # total_risk = steps.loc[:, 'absolute_risk'].sum(axis=0)
        # steps.loc[:, 'cummulative_risk'] = steps.loc[:, 'absolute_risk'].cumsum() / total_risk
        # steps.drop(['absolute_risk', 'most_risky', 'local_max_step'], axis=1, inplace=True)
        steps.drop(['most_risky', 'local_max_step'], axis=1, inplace=True)
# TO ENCREASE THE PERFORMANCE - COMMENT NEXT LINE
        steps = steps.sort_values('n_clusters', ascending=True)
        return steps

# ONE OTHER WAY TO EVALUATE THE BEST VALUE OF lambda FOR CLUSTER MERGE
    def two_distance_clusters(self):
        centroid_distances = self.stepwise_merge
        clusterer = KMeans(n_clusters=2, random_state=0).fit(pd.DataFrame(centroid_distances.loc[:, 'lambda']))
        centroid_distances.loc[:, 'label'] = clusterer.labels_
        return centroid_distances.groupby('label')['n_clusters'].max().min()


    def __init__(self, source_clustering=None, centroids=None, volumes=None, radii=None):
        if (source_clustering is None) and (centroids is None):
            pass
        else:
            if source_clustering is None:
                self.centroids = centroids
                if volumes is None:
                    self.volumes = pd.DataFrame(np.ones((self.centroids.shape[0], 2)), columns=['volume', 'fraction'])
                    self.volumes.index = range(self.centroids.shape[0])
                    self.volumes.loc[:, 'cluster'] = self.volumes.index
                else:
                    self.volumes = pd.DataFrame(volumes)
            else:
                self.source_clustering = source_clustering
                self.source_labels = source_clustering.labels_
                self.source_labels = pd.DataFrame(self.source_labels, columns=['source'])
                self.source_n_rows = self.source_labels.shape[0]
                self.volumes = self.get_volumes()
                self.centroids = source_clustering.cluster_centers_
            self.radii = radii
            if radii is not None:
                self.radii = pd.DataFrame(self.radii)
                self.radii.index = self.radii.loc[:, 'cluster']
                self.radii.drop(['cluster'], axis=1, inplace=True)
                self.radii.columns = ['radius']
            self.centroids = pd.DataFrame(self.centroids)
            self.source_n_clusters = self.centroids.shape[0]
            self.centroid_distances = self.get_centroid_distances()
            self.stepwise_merge, self.stepwise_label_map = self.cluster_merge('lambda')
            # print(self.stepwise_merge)
            # print(self.stepwise_label_map)
            self.k_between_two_distance_clusters = self.two_distance_clusters()
            self.steps = self.measure_steps('lambda')
            self.n_merged = None

    def get_label_map(self, n_merged):
        self.n_merged = n_merged
        self.label_map = self.stepwise_label_map[n_merged]
        return self.label_map

    def predict(self, n_merged=None, some_dataframe=None):
        if some_dataframe is None:
            source_prediction = self.source_labels
        else:
            # source_prediction = pd.DataFrame(self.source_clustering.predict(some_dataframe), columns=['source']).set_index('source')
            source_prediction = pd.DataFrame(self.source_clustering.predict(some_dataframe), columns=['source'])
        if (n_merged is None) & (self.n_merged is None):
            return source_prediction
        else:
            if (n_merged is None):
                n_merged = self.n_merged
            # label_map = self.stepwise_label_map[n_merged].set_index('source')
            label_map = self.stepwise_label_map[n_merged]
            # return source_prediction.join(label_map)
            return pd.merge(source_prediction, label_map, on="source", how='left')

    def control_set(self, source_set, n_select, n_merged=None, random_state=0, pattern_set=None):
        if (n_merged is None) & (self.n_merged is None):
            n_merged = self.source_n_clusters
        else:
            if (n_merged is None):
                n_merged = self.n_merged
        source_merged_labels = self.predict(n_merged)
        if pattern_set is None:
            merged_labels = source_merged_labels
        else:
            merged_labels = predict(n_merged, pattern_set)
        merged_volumes = self.get_volumes(merged_labels, 'merged')
        merged_volumes.loc[:, 'n_select'] = merged_volumes.loc[:, 'fraction'] * n_select
        merged_volumes.loc[:, 'n_select'] = merged_volumes.loc[:, 'n_select'].round(0)

        selected_set = pd.DataFrame()
        for every_label in merged_volumes['cluster']:
            selected_set = pd.concat([
                selected_set,
                source_set.loc[source_merged_labels == every_label, :].sample(
                    n=merged_volumes.loc[every_label, 'n_select'],
                    random_state=random_state
                )
            ])
        return selected_set

if __name__ == '__main__':
    pass



# PLOT DEPENDENCY BETWEEN lambda AND n_clusters - RESULTS OF MERGE
# import matplotlib.pyplot as plt
# plt.plot(lambda.loc[:, 'lambda'], lambda.loc[:, 'n_clusters'], marker='s')
