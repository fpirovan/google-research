# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import tempfile
from absl.testing import absltest
import numpy as np

from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind


class ScannTest(absltest.TestCase):

  def verify_serialization(self, ds, searcher, n_dims, n_queries):
    with tempfile.TemporaryDirectory() as tmpdir:
      searcher.serialize(tmpdir)
      queries = []
      indices = []
      distances = []
      for _ in range(n_queries):
        queries.append(np.random.rand(n_dims).astype(np.float))
        idx_orig, dis_orig = searcher.search(queries[-1])
        indices.append(idx_orig)
        distances.append(dis_orig)

      s2 = scann_ops_pybind.load_searcher(ds, tmpdir)
      for q, idx, dis in zip(queries, indices, distances):
        idx_new, dis_new = s2.search(q)
        np.testing.assert_array_equal(idx_new, idx)
        np.testing.assert_allclose(dis_new, dis)

  def normalize(self, dataset):
    return dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

  def test_brute_force(self):

    def ground_truth(dataset, query, k):
      product = np.matmul(dataset, query)
      idx = np.argsort(product)[-k:]
      dis = np.sort(product)[-k:]
      return idx[::-1], dis[::-1]

    k = 10
    n_dims = 10
    n_points = 1000
    ds = np.random.rand(n_points, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(
        ds, k, "dot_product").score_brute_force().create_pybind()
    for _ in range(100):
      q = np.random.rand(n_dims).astype(np.float)
      idx, dis = s.search(q)
      gt_idx, gt_dis = ground_truth(ds, q, 10)

      np.testing.assert_array_equal(idx, gt_idx)
      np.testing.assert_allclose(dis, gt_dis, rtol=1e-5)

  def test_batching(self):
    k = 10
    n_dims = 10
    n_points = 1000
    ds = np.random.rand(n_points, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(
        ds, k, "dot_product").score_brute_force().create_pybind()

    qs = np.random.rand(n_points, n_dims).astype(np.float)
    batch_idx, batch_dis = s.search_batched(qs)
    for i, q in enumerate(qs):
      idx, dis = s.search(q)

      np.testing.assert_array_equal(idx, batch_idx[i])
      np.testing.assert_allclose(dis, batch_dis[i], rtol=1e-5)

  def test_tree_ah(self):
    n_dims = 50
    n_points = 10000
    ds = np.random.rand(n_points, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(ds, 10, "dot_product").tree(
        300, 30, min_partition_size=10).score_ah(2).create_pybind()
    self.verify_serialization(ds, s, n_dims, 5)

  def test_pure_ah(self):
    n_dims = 50
    n_points = 10000
    ds = np.random.rand(n_points, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(ds, 10,
                                   "dot_product").score_ah(2).create_pybind()
    self.verify_serialization(ds, s, n_dims, 5)

  def test_tree_brute_force(self):
    n_dims = 100
    ds = np.random.rand(12345, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(ds, 10, "dot_product").tree(
        100, 10).score_brute_force(True).create_pybind()
    self.verify_serialization(ds, s, n_dims, 5)

  def test_shapes(self):
    n_dims = 128
    k = 10
    ds = np.random.rand(1234, n_dims).astype(np.float)
    # first look at AH searcher with reordering
    s = scann_builder.ScannBuilder(
        ds, k, "dot_product").score_ah(2).reorder(30).create_pybind()
    q = np.random.rand(n_dims).astype(np.float)
    self.assertLen(s.search(q)[0], k)
    self.assertLen(s.search(q, final_num_neighbors=20)[0], 20)
    self.assertLen(s.search(q, pre_reorder_num_neighbors=50)[0], k)
    self.assertLen(
        s.search(q, final_num_neighbors=15, pre_reorder_num_neighbors=30)[0],
        15)

    batch_q = np.random.rand(18, n_dims).astype(np.float)
    self.assertEqual(s.search_batched(batch_q)[0].shape, (18, k))
    self.assertEqual(
        s.search_batched(batch_q, final_num_neighbors=20)[0].shape, (18, 20))
    self.assertEqual(
        s.search_batched(batch_q, pre_reorder_num_neighbors=20)[0].shape,
        (18, k))
    self.assertEqual(
        s.search_batched(
            batch_q, final_num_neighbors=20,
            pre_reorder_num_neighbors=40)[0].shape, (18, 20))

    # now look at AH without reordering
    s2 = scann_builder.ScannBuilder(ds, k,
                                    "dot_product").score_ah(2).create_pybind()
    self.assertLen(s2.search(q)[0], k)
    self.assertLen(s2.search(q, final_num_neighbors=20)[0], 20)

  def test_squared_l2(self):

    def ground_truth(dataset, query, k):
      squared_l2 = np.sum(np.square(dataset - query), axis=1)
      return np.argsort(squared_l2)[:k], np.sort(squared_l2)[:k]

    n_dims = 50
    k = 10
    ds = np.random.rand(2500, n_dims).astype(np.float)
    qs = np.random.rand(500, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(
        ds, k, "squared_l2").score_brute_force(False).create_pybind()
    idx, dis = s.search_batched(qs)

    for query, idx_row, dis_row in zip(qs, idx, dis):
      _, gt_dis = ground_truth(ds, query, k)
      np.testing.assert_allclose(dis_row, gt_dis, rtol=1e-5)
      selected_distances = np.sum(np.square(ds[idx_row] - query), axis=1)
      np.testing.assert_allclose(dis_row, selected_distances, rtol=1e-5)

  def test_parallel_batching(self):
    n_dims = 50
    k = 10
    ds = np.random.rand(12500, n_dims).astype(np.float)
    qs = np.random.rand(2500, n_dims).astype(np.float)
    s = scann_builder.ScannBuilder(ds, k, "squared_l2").tree(
        80, 10).score_ah(2).create_pybind()
    idx, dis = s.search_batched(qs)
    idx_parallel, dis_parallel = s.search_batched_parallel(qs)
    np.testing.assert_array_equal(idx, idx_parallel)
    np.testing.assert_array_equal(dis, dis_parallel)

  # make sure spherical partitioning proto is valid and doesn't crash
  def test_spherical_kmeans(self):
    n_dims = 50
    k = 10
    ds = self.normalize(np.random.rand(12500, n_dims).astype(np.float))
    s = scann_builder.ScannBuilder(ds, k, "squared_l2").tree(
        80, 10, spherical=True).score_ah(2).create_pybind()
    self.verify_serialization(ds, s, n_dims, 20)


if __name__ == "__main__":
  absltest.main()