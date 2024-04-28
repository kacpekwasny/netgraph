import numpy
import tensorflow as tf
import spektral

cora = spektral.datasets.Citation(name='cora')

test_mask = cora.mask_te
test_train = cora.mask_tr
test_valid = cora.mask_va
graph = cora.graphs[0]
features = graph.x
adj = graph.a.todense()
labels = graph.y

features = features.todense()

