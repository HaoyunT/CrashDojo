"""Geometry regression checks for failures that center/endpoint checks miss."""
import numpy as np
from corridor_prefix_gate import make_corridor, prefix_coverages

identity = np.eye(3)
lane = [[-20, -1], [20, -1], [20, 1], [-20, 1], [-20, -1]]
corridor = make_corridor([lane])
p = np.array([[[.1, 0]], [[.1, .2]]])
y = np.zeros((2, 1))
coverage = prefix_coverages(p, y, identity, [4, 2], corridor)
assert np.allclose(coverage, [1., .9], atol=1e-8), coverage

# Both endpoint boxes fit in the union, but the path traverses a corridor gap.
left = [[-3, -1], [.7, -1], [.7, 1], [-3, 1], [-3, -1]]
right = [[1.3, -1], [4, -1], [4, 1], [1.3, 1], [1.3, -1]]
gap = make_corridor([left, right])
coverage = prefix_coverages(np.array([[[2., 0.]]]), np.zeros((1, 1)),
                           identity, [.2, .2], gap)
assert coverage[0] < 1e-8, coverage

# Same centerline, body rotation crosses the lane edge.
rotated = prefix_coverages(np.array([[[.1, 0.]]]), np.array([[.2]]),
                           identity, [4, 2], corridor)
assert rotated[0] < .95, rotated

# World coordinate translation/rotation preserves the result.
theta = .6
tf = np.eye(3)
tf[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
tf[:2, 2] = [1292., 1057.]
world_lane = np.asarray(lane) @ tf[:2, :2].T + tf[:2, 2]
transformed = prefix_coverages(p, y, tf, [4, 2], make_corridor([world_lane]))
assert np.allclose(transformed, [1., .9], atol=1e-8), transformed

bad = prefix_coverages(np.array([[[30., 0.]], [[np.nan, 0.]]]),
                       np.zeros((2, 1)), identity, [4, 2], corridor)
assert np.all(bad == 0), bad
print("PASS: body area, between-frame gap, rotation, world transform, invalid jumps")
