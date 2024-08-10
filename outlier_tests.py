from utils import *
from ploting import *
from outlier import add_outlier_to_pc
from sampling_methods import *
from frequency_domain_oprations import freq_based_sampling
pc = read_point_cloud("/home/behnam/phd/research/convex hull/modelnet40_frames/airplane_0001.txt")
pc = pc[:,0:3]
pc = pc[::10]
pc = pc_normalize(pc)

pc = add_outlier_to_pc(pc, 0)
low_freq_rec = freq_based_sampling(pc)

#whole = np.concatenate((pc, low_freq_rec))
#plot_style_2(whole)
#min_distances, arg_min_distances = find_distance_pc(pc, low_freq_rec)
indices = [0]
#plot_style_2(pc, [0])
#plot_style_2(low_freq_rec, [297])

fps = farthest_point_sample(pc, 30)


plot_style_2(low_freq_rec)


