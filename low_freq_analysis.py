#from __future__ import division
from ploting import *
from sampling_methods import *
from frequency_domain_oprations import *
from utils import *
import numpy as np
import time


pc = read_point_cloud("/home/behnam/phd/research/convex hull/modelnet40_frames/airplane_0001.txt")

pc = pc[:,0:3]
pc = pc[::10]
pc = pc_normalize(pc)
grid_pc = make_full_pc(pc)

npoint = 64
proposed_time = -time.time()
fps_proposed = farthest_point_sample_proposed(np.expand_dims(pc, 0), npoint).astype(np.int32)
proposed_time += time.time()
orig_time = -time.time()
fps_orig = farthest_point_sample(np.expand_dims(pc,0), npoint)
orig_time += time.time()
random_time = -time.time()
rps = random_sampling(np.expand_dims(pc, axis=0), npoint).astype(np.int32)
random_time += time.time()


coverage_rps = compute_coverage(np.expand_dims(grid_pc, axis = 0), np.expand_dims(pc, axis = 0), rps)
coverage_fps_orig = compute_coverage(np.expand_dims(grid_pc, axis = 0), np.expand_dims(pc, axis = 0), fps_orig)
coverage_fps_proposed = compute_coverage(np.expand_dims(grid_pc, axis = 0), np.expand_dims(pc, axis = 0), fps_proposed)

#test_plot2(pc, fps_orig[0,:])
#test_plot2(pc, rps[0,:])
#test_plot2(pc, fps_proposed[0,:])


print("Times: ")
print(orig_time)
print(proposed_time)
print(random_time)
print(len(coverage_fps_orig))
print(len(coverage_fps_proposed))
print(len(coverage_rps))
exit(0)
#test_plot2(coverage[0,:,:])
#whole = np.concatenate((pc, grid_pc))
#test_plot2(whole)
#print(compute_coverage(pc))
#print(compute_coverage(fps1))
#print(compute_coverage(fps2))


#test_plot2(pc)
exit(0)




test_plot2(pc)
plot_pc(pc)



corrupted = add_outlier_to_pc(pc, 12)

smooth_pc = freq_based_sampling(pc)
smooth_pc = pc_normalize(smooth_pc)


smooth_corrupted = freq_based_sampling(corrupted)
smooth_corrupted = pc_normalize(smooth_corrupted)



#plot_pc(pc)
#plot_pc(corrupted)
#plot_pc(smooth_pc)
#plot_pc(smooth_corrupted)


fps = farthest_point_sample_orig(np.expand_dims(corrupted, 0), 128)
fps = np.sort(fps)

exit(0)

print("--------------------")
print(pc_bounds(pc))
pc = pc_normalize(pc)
print(pc_bounds(pc))
exit(0)

print(min(pc[:,0]))
print(max(pc[:,0]))



#plot_pc(pc)
smooth_pc = freq_based_sampling(pc)
smooth_pc = pc_normalize(smooth_pc)
whole = np.concatenate((pc, smooth_pc))
fps = farthest_point_sample_orig(np.expand_dims(pc, 0), 396)
rps = random_point_sample(pc, 396)
fps = fps[0,:]
rps = rps[0,:]

#plot_pc(whole)



print(min(smooth_pc[:,0]))
print(max(smooth_pc[:,0]))

print(calculate_min_distance(pc, rps))
print(calculate_min_distance(pc, fps))
#print(calculate_pc_min_distance(pc))
print(calculate_pc_min_distance(smooth_pc))
