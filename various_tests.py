pc = read_point_cloud("/home/behnam/phd/research/convex hull/modelnet40_frames/airplane_0001.txt")
pc = pc[:,0:3]
pc = pc[0:1024]



pc2 = make_target_frame(pc)
whole = np.concatenate((pc, pc2))
plot_style_2(whole, list(range(1024,2048)))
result = register(pc, pc2)

distances = sklearn.metrics.pairwise.euclidean_distances(pc2,result)
score = np.sum(np.min(distances, axis = 0))
print("score: ", score)

whole2 = np.concatenate((result, pc2))
plot_style_2(whole2)


#plot_style_2(pc2)
exit(0)




initial_point = 1024
train_data, train_label = data_load(num_point=initial_point,
                                               data_dir='/home/behnam/phd/research/frequency_domain/temp_test', train=True)

pc = train_data[17, :, :]
#pc = pc_normalize(pc)
#pc2 = train_label[0, :, :]
plot_style_2(pc)
#plot_style_2(pc2)

f = h5py.File('/home/behnam/phd/research/frequency_domain/modelnet40_ply_hdf5_2048/ply_data_test0.h5', 'r')




# Tests on Registration
#pc = read_point_cloud("/home/behnam/phd/research/convex hull/modelnet40_frames/person_0003.txt")
pc = np.load("temp_pc.npy")
pc2 = np.load("temp_pc2.npy")
# pc = np.load("/home/behnam/temp/R-PointHop/source_0.ply.npy")
# pc2 = np.load("/home/behnam/temp/R-PointHop/target_0.ply.npy")
# pc = pc_normalize(pc)
# pc2 = pc_normalize(pc2)
# pc = add_outlier_to_pc(pc, 10)[10:]
# pc2 = (add_outlier_to_pc(pc2 , 10))[10:]
# np.save("temp_pc.npy", pc)
# np.save("temp_pc2.npy", pc2)
# exit(0)

register_time = -time.time()
result = register(pc, pc2)
register_time += time.time()
print("time: ", register_time)
distances = sklearn.metrics.pairwise.euclidean_distances(pc2,result)
score = np.sum(np.min(distances, axis = 0))
print("score: ", score)
whole = np.concatenate((pc2, result))
plot_style_2(whole, list(range(1024, 2048)))
exit(0)


# Temp
pc2 = add_outlier_to_pc(pc2, 10)

fps_proposed = distance_aware_farthest_point_sample(pc2, 64)
#fps_orig = farthest_point_sample(pc2, 640)
plot_style_2(pc2, fps_proposed)


#pc = add_outlier_to_pc(pc, 30)
#pc2 = add_outlier_to_pc(pc2, 30)
result = register(pc, pc2)
exit(0)
dual_result = np.concatenate((pc2, result))
plot_style_2(dual_result, list(range(1024,2048)))

#pc = add_outlier_to_pc(pc, 10)
#pc2 = add_outlier_to_pc(pc2,10)

#np.save("/home/behnam/temp/R-PointHop/source_0_corrupted.ply.npy", pc[10:])
#np.save("/home/behnam/temp/R-PointHop/target_0_corrupted.ply.npy", pc2[10:])
#exit(0)


#pc3 = np.load("/home/behnam/temp/R-PointHop/result.npy")
#pc2 = np.concatenate((pc2, pc3))
plot_style_2(pc2, list(range(1024,2048)))
plot_style_2(pc3)
exit(0)
pc = pc[:,0:3]
pc = pc[::10]
pc = pc_normalize(pc)

pc_corrupted = add_outlier_to_pc(pc, 10)

fps_time = -time.time()
fps_orig = farthest_point_sample(pc_corrupted, 8)
fps_time += time.time()


fps_proposed_time = -time.time()
fps_proposed = farthest_point_sample_proposed(pc, 8)
fps_proposed_time += time.time()



low_pass_time = -time.time()
low_freq_rec = freq_based_sampling(pc_corrupted)
low_freq_rec = pc_normalize(low_freq_rec)
low_pass_time += time.time()


min_distances, arg_min_distances = find_distance_pc(pc_corrupted, low_freq_rec)

npoint = 30


fps_proposed = distance_aware_farthest_point_sample(pc_corrupted, 30, min_distances)

whole = np.concatenate((pc_corrupted, low_freq_rec))
plot_style_2(pc_corrupted)
plot_style_2(pc_corrupted, fps_orig)
plot_style_2(pc_corrupted, fps_proposed)
plot_style_2(whole)

A = 9



#whole = np.concatenate((pc, low_freq_rec))
#plot_style_2(whole)

#indices = [0]
#plot_style_2(pc, [0])
#plot_style_2(low_freq_rec, [297])

#fps = farthest_point_sample(pc, 30)


#plot_style_2(low_freq_rec)

