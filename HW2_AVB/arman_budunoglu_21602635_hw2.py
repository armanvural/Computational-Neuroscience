import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import os
import scipy.signal



question = input("Enter Relevant Question Number to Display its Respective Output \n")

def arman_budunoglu_21602635_hw2(question):

    if question == "1":

        #q1_plot_directory = ['Q1|A', 'Q1|B', 'Q1|C']

        print("Question 1: ")

        # Loading the dataset
        dataset = sc.loadmat('c2p3.mat')
        # Examining its contents
        print("Contents of c2p3.mat file: " + str(dataset.keys()))


        print("Question 1: Part A")
        # Inspecting counts array
        counts = dataset['counts']
        print('counts is of shape: ' + str(counts.shape)) # (32767, 1)
        # Inspecting stim
        stim = dataset['stim']
        print('stim is of shape: ' + str(stim.shape))  # (16, 16, 32767)

        # Using the STA_vals function for generating 10 different time step STAs
        # Function returns a list containing STAs indexed by their step size-1
        STA_steps = STA_vals(counts,stim)


        # Plotting each STA as an image
        for t in range(10):
            STA = STA_steps[t]
            plt.figure()
            plt.imshow(STA, cmap='gray', vmin=np.min(STA_steps), vmax=np.max(STA_steps))
            plt.title('STA :' + str(t+1) + ' Step Prior to Spike')
            #plt.savefig(q1_plot_directory[0]+"/Q1|PART:A STA" +str(t+1)+ ".png",bbox_inches='tight')
            plt.show()


        print("Question 1: Part B")

        # Generating a 3 dimensional 16x16x10 matrix containing all STDs
        STD_kernel = np.zeros((16,16,10))
        # Plugging STDs to the generated matrix
        for i in range(10):
            STD_kernel[:,:,i] = STA_steps[i]

        # Summing over one spatial-dimension
        STA_rows = np.sum(STD_kernel, axis=0)

        # Displaying the resulting matrix as an image
        plt.figure()
        plt.imshow(STA_rows, cmap='gray')
        plt.title('STAs Summed Along the Rows')
        #plt.savefig(q1_plot_directory[1]+"/Q1|PART:B.png",bbox_inches='tight')
        plt.show()


        print("Question 1: Part C")
        # Single time step STA image
        STA_1 = STA_steps[0]

        # Obtaining projections for each of the 32767 sample stimuli
        projections_all = project_sample(stim,STA_1)

        # Normalizing the returned array containing projections
        norm_projections = projections_all/np.max(projections_all)
        # Using identical bin size
        hist_bin = 100
        # Plotting All Stimulus Projections Histogram
        plt.figure()
        plt.hist(norm_projections, bins=hist_bin)
        plt.title('All Stimulus Projections Histogram')
        plt.xlabel('Projection')
        plt.ylabel('Count')
        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C.png", bbox_inches='tight')
        plt.show()

        # Repeating the same procedure for non-zero count inducing stimuli
        projections_nz = non_zero_projections(counts,stim,STA_1)
        # Normalizing the returned array containing projections
        norm_nz_projections = projections_nz / np.max(projections_nz)
        # Plotting Non-zero Spike Inducing Stimulus Samples Projections Histogram
        plt.figure()
        plt.hist(norm_nz_projections, bins=hist_bin)
        plt.title('Non-Zero Spike Inducing Stimulus Projections Histogram')
        plt.xlabel('Projection')
        plt.ylabel('Count')
        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C_2.png", bbox_inches='tight')
        plt.show()

        # Plotting both histograms in a single graph for comparison
        plt.figure()
        all_projections_hist = plt.hist(norm_projections, bins=hist_bin, color='black', alpha=0.75, label='All Samples',rwidth=0.7)
        nz_projections_hist = plt.hist(norm_nz_projections, bins=hist_bin, color='red', alpha=0.75, label='Non-Zero Spike Samples',rwidth=0.7)
        plt.title('All vs Spike Inducing Stimulus Projections on STA Image')
        plt.xlabel("Projection")
        plt.ylabel("Counts")
        plt.legend()
        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C_3.png", bbox_inches='tight')
        plt.show()





    elif question == "2":
        #q2_plot_directory = ['Q2|A', 'Q2|B', 'Q2|C', 'Q2|D', 'Q2|E', 'Q2|F']

        print("Question 2")

        print("Question 2: Part A")

        # Sampling the Receptive Field as a 21x21 Matrix
        dog_field_sample = np.zeros((21, 21))
        # The field is sampled for x,y values between -10 and 10
        for i in range(21):
            for j in range(21):
                x_axis = i - 10
                y_axis = j - 10
                dog_field_sample[i][j] = DOG_field(2, 4, x_axis, y_axis)

        # Displaying the resulting sample image
        plt.figure()
        plt.imshow(dog_field_sample)
        plt.title('Sampled DOG Receptive Field Image')
        #plt.savefig(q2_plot_directory[0] + "/Q2|PART:A.png", bbox_inches='tight')
        plt.show()


        print("Question 2: Part B")

        # Loading the given image sample
        visual_stim = plt.imread('hw2_image.bmp')
        print('hw2_image dimensions : ' + str(visual_stim.shape)) # (480, 512, 3)

        # Convolving sampled DOG field with the given image for obtaining LGN cells response
        response = scipy.signal.convolve(visual_stim[:,:,0],dog_field_sample, mode='same')

        # Displaying the response as an image along with the original image
        plt.figure()
        plt.imshow(response, cmap='gray')
        plt.title('Neural Activity Image')
        #plt.savefig(q2_plot_directory[1] + "/Q2|PART_B_Response.png", bbox_inches='tight')
        plt.show()
        plt.figure()
        plt.imshow(visual_stim)
        plt.title('Original Image')
        #plt.savefig(q2_plot_directory[1] + "/Q2|PART_B_OG.png", bbox_inches='tight')
        plt.show()


        print("Question 2: Part C")

        # Trying several threshold values
        th_list = list()
        for t in range(-2, 4):
            detection_img = edge_detect(response, t)
            th_list.append(detection_img)

        # Plotting them together with subplot function
        t_plots = plt.figure()
        row = 3
        col = 3
        ax = []
        for i in range(len(th_list)):
            ax.append(t_plots.add_subplot(row, col, i + 1))
            title = ("Thresholding with: " + str(i - 2))
            ax[-1].set_title(title)
            plt.imshow(th_list[i], cmap='gray')

        t_plots.tight_layout()
        #plt.savefig(q2_plot_directory[2] + "/Q2|C_subplots.png", bbox_inches='tight')
        plt.show()

        # Plotting optimized detector output
        plt.figure()
        plt.imshow(th_list[2],cmap='gray')
        plt.title('Edge Detector Output with Threshold = 0')
        #plt.savefig(q2_plot_directory[2] + "/Q2|C.png", bbox_inches='tight')
        plt.show()


        print("Question 2: Part D")
        # Initializing parameters # theta = np.pi / 2 # std_l = 3 # std_w = 3 # lamb = 6
        # phi = 0

        # Generating the Gabor field with given specifications
        G_rec_field = np.zeros((21, 21))
        G_zero = np.zeros((21, 21))
        for i in range(21):
            for j in range(21):
                x_ax = i - 10
                y_ax = j - 10
                G_rec_field[i][j] = Gabor_field(np.pi/2, 0, 6, 3, 3, x_ax, y_ax)
                G_zero[i][j] = Gabor_field(0, 0, 6, 3, 3, x_ax, y_ax)

        # Displaying the obtained Gabor Receptive Field Image
        plt.figure()
        plt.imshow(G_rec_field)
        plt.title('Gabor Receptive Field Constructed with ' r'$\theta$ = 90')
        #plt.savefig(q2_plot_directory[3] + "/Q2|D.png", bbox_inches='tight')
        plt.show()

        print("Question 2: Part E")
        # Computing the Responses of V1 neurons to given image theta = 90
        Gabor_90_resp = scipy.signal.convolve(visual_stim[:,:,0],G_rec_field, mode='same')
        # Theta = 0
        Gabor_0_resp = scipy.signal.convolve(visual_stim[:,:,0],G_zero, mode='same')

        #  Displaying Response Image obtained with Gabor 90
        plt.figure()
        plt.imshow(Gabor_90_resp, cmap='gray')
        plt.title(' Neural Activity Image with Gabor Orientation ' r'$\theta$ = 90')
        #plt.savefig(q2_plot_directory[4] + "/Q2|E_90.png", bbox_inches='tight')
        plt.show()

        gabor_list = list()
        gabor_list.append(Gabor_90_resp)
        gabor_list.append(Gabor_0_resp)
        gabor_90_edges = edge_detect(Gabor_90_resp,0)
        gabor_0_edges = edge_detect(Gabor_0_resp, 0)
        gabor_list.append(gabor_90_edges)
        gabor_list.append(gabor_0_edges)
        orions_belt = ['90', '0','90', '0']

        g_plots = plt.figure()
        row = 2
        col = 2
        axe = []
        for i in range(len(gabor_list)):
            axe.append(g_plots.add_subplot(row, col, i + 1))
            title = ('Gabor Orientation ' r'$\theta$ = ' + str(orions_belt[i]))
            axe[-1].set_title(title)
            plt.imshow(gabor_list[i], cmap='gray')

        g_plots.tight_layout()
        #plt.savefig(q2_plot_directory[4] + "/Q2|E_subplots.png", bbox_inches='tight')
        plt.show()

        print("Question 2: Part F")

        # Constructing 2 more Gabors with 30 and 60 degree orientations
        Gabor_30 = np.zeros((21, 21))
        Gabor_60 = np.zeros((21, 21))
        for i in range(21):
            for j in range(21):
                x_ax = i - 10
                y_ax = j - 10
                Gabor_30[i][j] = Gabor_field(np.pi/6, 0, 6, 3, 3, x_ax, y_ax)
                Gabor_60[i][j] = Gabor_field(np.pi/3, 0, 6, 3, 3, x_ax, y_ax)

        # Obtaining their individual responses
        Gabor_30_resp = scipy.signal.convolve(visual_stim[:,:,0],Gabor_30, mode='same')
        # Theta = 0
        Gabor_60_resp = scipy.signal.convolve(visual_stim[:,:,0],Gabor_60, mode='same')

        # Generating a list for all Gabor receptive field constructs
        gabor_constructs = [G_zero, Gabor_30, Gabor_60,G_rec_field]
        # Generating a list for all Gabor responses
        gabor_responses = [Gabor_0_resp, Gabor_30_resp, Gabor_60_resp, Gabor_90_resp]

        # Displaying all constructs and responses with 2 subplots
        theta_params = ['0', '30', '60', '90']
        constructs = plt.figure()
        row = 2
        col = 2
        cons_ax = []
        for i in range(len(gabor_constructs)):
            cons_ax.append(constructs.add_subplot(row, col, i + 1))
            title_c = ('Gabor Field with ' r'$\theta$ = ' + str(theta_params[i]))
            cons_ax[-1].set_title(title_c)
            plt.imshow(gabor_constructs[i])

        constructs.tight_layout()
        #plt.savefig(q2_plot_directory[5] + "/Q2|F_fields.png", bbox_inches='tight')
        plt.show()

        responses = plt.figure()
        resp_ax = []
        for j in range(len(gabor_responses)):
            resp_ax.append(responses.add_subplot(row, col, j + 1))
            title_s = ('Response for Gabor ' r'$\theta$ = ' + str(theta_params[j]))
            resp_ax[-1].set_title(title_s)
            plt.imshow(gabor_responses[j], cmap='gray')


        responses.tight_layout()
        #plt.savefig(q2_plot_directory[5] + "/Q2|F_resps.png", bbox_inches='tight')
        plt.show()

        # Combined Response
        combined_response = Gabor_0_resp + Gabor_30_resp + Gabor_60_resp + Gabor_90_resp
        plt.figure()
        plt.imshow(combined_response, cmap='gray')
        plt.title('Combined Neural Response')
        #plt.savefig(q2_plot_directory[5] + "/Q2|F_COMB.png", bbox_inches='tight')
        plt.show()




# Functions used in the assignment

# Obtaining STA values for Question 1 part A
def STA_vals(spike_count, stimulus):
    # Initializing parameters
    time_step = 1
    total_spikes = 0
    # Reshaping stim
    stims = stimulus.T
    # Initializing a list to store STAs for each 10 steps before spikes
    STA_list = list()
    # Obtaining total spike count sum for averaging
    spike_sum = np.sum(spike_count)

    # 10 iterations for getting 10 time steps of STAs
    while time_step < 11:
        # Initializing parameters
        index_count = -1
        STA = np.zeros((16, 16))

        for a in spike_count:
            index_count += 1
            if a != 0:
                if index_count-time_step > 0:
                    STA += stims[index_count-time_step]*a

        # Obtaining the average
        STA_step = STA/spike_sum
        # Appending 10 different time step STA's indexed respectively
        STA_list.append(STA_step)
        time_step += 1
    return STA_list

# Question 1 Part C
#  Frobenius Inner Product computation for all time samples

def project_sample(stimuli, STA_sing):
    sample_size = 32767
    # Initializing an array for storing each projection
    sample_proj = np.zeros((32767))
    stims = stimuli.T

    for t in range(sample_size):
        sample_proj[t] = np.sum(stims[t] * STA_sing)

    return sample_proj

#  Frobenius Inner Product computation for non-zero spike count inducing stimuli samples
def non_zero_projections(spike_count, stimuli, STA_sing):
    # Obtaining the non-zero indexes
    nz_indexes = np.nonzero(spike_count)[0]
    #  Obtaining number of non-zero spike instances
    nz_spikes = np.count_nonzero(spike_count)
    #  Initializing an empty array of size non-zero counts
    nz_projections = np.zeros((nz_spikes))

    stims = stimuli.T
    for t in range(nz_spikes):
        nz_projections[t] = np.sum(stims[nz_indexes[t]] * STA_sing)

    return nz_projections

# Functions used in Question 2

# Defining the DOG receptive field function
def DOG_field(std_c, std_s, x, y):
    central_G = (1 / (2 * np.pi * std_c ** 2)) * (np.exp(-(x ** 2 + y ** 2) / (2 * std_c ** 2)))
    surr_G = (1 / (2 * np.pi * std_s ** 2)) * (np.exp(-(x ** 2 + y ** 2) / (2 * std_s ** 2)))
    dog = central_G - surr_G

    return dog

# Edge Detection Function utilized in Q2:C
def edge_detect(image, th_val):
    th_image = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > th_val:
                th_image[i][j] = 1
            else:
                th_image[i][j] = 0

    return th_image

# Defining the Gabor receptive field
def Gabor_field(theta, phi, lambda_param, std_l, std_w, x_param, y_param):
    # Defining k unit vectors
    k = np.array([np.cos(theta), np.sin(theta)])
    k_ort = np.array([-np.sin(theta), np.cos(theta)])
    # dot products
    k_x = k[0] * x_param + k[1] * y_param
    k_orth_x = k_ort[0] * x_param + k_ort[1] * y_param
    # Gaussian part of the expression
    Gauss = np.exp(-((k_x ** 2) / (2 * (std_l ** 2))) - ((k_orth_x ** 2) / (2 * (std_w ** 2))))
    #  cosine part
    cos = np.cos((2 * np.pi * k_orth_x / lambda_param) + phi)
    G_rec_field = Gauss * cos

    return G_rec_field



arman_budunoglu_21602635_hw2(question)



