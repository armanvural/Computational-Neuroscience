import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF

question = input("Enter Relevant Question Number to Display its Respective Output \n")

def arman_budunoglu_21602635_hw4(question):

    if question == "1":

        q1_plot_directory = ['Q1|A', 'Q1|B', 'Q1|C', 'Q1|D']



        print("Question 1: ")

        # Loading the dataset
        hf = h5py.File('hw4_data1.mat', 'r')
        ls = list(hf.keys())
        # Examining its contents
        print("Contents of hw4_data1.mat file: " + str(ls))


        print("Question 1: Part A")
        # Inspecting faces array
        faces = np.array(hf.get('faces'))
        print('faces is of shape: ' + str(faces.shape)) # (1024, 1000)

        # Transposing the dataset
        faces = faces.T # (1000, 1024)
        # Normalizing the data
        # Mean of every feature column
        f_mean = np.mean(faces,axis=0)
        # Subtracting the mean from data
        faces_f = faces - f_mean
        # Obtaining the variance of the features
        var_f = np.var(faces_f, axis=0)
        std_f = np.sqrt(var_f)
        # Normalized data obtained with
        norm_faces = faces_f/std_f

        # First 100 Principal Components
        pca_data = PCA(100)
        # Obtaining the PCA output
        pca_a = pca_data.fit(norm_faces)

        # Obtaining PVE
        pve_a = pca_data.explained_variance_ratio_
        # Plotting 100 PCs PVE
        plt.figure()
        plt.plot(pve_a)
        plt.title('First 100 PCs PVE')
        plt.xlabel('PC Index')
        plt.ylabel('PVE of Individual PC')
        #plt.savefig(q1_plot_directory[0] + "/Q1|PART:A_PVE.png", bbox_inches='tight')
        plt.show()

        # Displaying first 25 PCs as Images
        pc_ims = plt.figure()
        row = 5
        col = 5
        ax = []
        for i in range(25):
            ax.append(pc_ims.add_subplot(row,col,i+1))
            plt.imshow(pca_a.components_[i].reshape(32,32).T, cmap='gray')
            ax[-1].tick_params(axis='both', labelsize=0, length=0)
            ax[-1].set_xlabel(i + 1)

        pc_ims.suptitle('First 25 PC Images')
        pc_ims.tight_layout()

        #plt.savefig(q1_plot_directory[0] + "/Q1|PART:A_PCA.png", bbox_inches=0)
        plt.show()




        print("Question 1: Part B")


        # Performing varying k-dimensional PCA applications
        k_10 = PCA(10)
        pca_10 = k_10.fit(norm_faces).components_
        k_25 = PCA(25)
        pca_25 = k_25.fit(norm_faces).components_
        k_50 = PCA(50)
        pca_50 = k_50.fit(norm_faces).components_

        # Projecting data onto each of the subspaces
        z_10 = np.dot(norm_faces, pca_10.T)
        z_25 = np.dot(norm_faces, pca_25.T)
        z_50 = np.dot(norm_faces, pca_50.T)

        # Reconstructing the data matrix from 3 projection cases
        x_10 = (np.dot(z_10,pca_10) * std_f) + f_mean
        x_25 = (np.dot(z_25, pca_25) * std_f) + f_mean
        x_50 = (np.dot(z_50, pca_50) * std_f) + f_mean


        # Plotting the Original first 36 Images
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row,col,i+1))
            plt.imshow(faces[i,:].reshape(32,32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('First 36 Original Images', y=1.0)
        ims.tight_layout(pad=0,w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[1] + "/Q1|PART:B_OG.png", bbox_inches='tight')
        plt.show()

        # Repeating the plotting algorithm for 10,25,50 PC projection reconstructions

        # Plotting k = 10 Reconstructed 36 Images
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row,col,i+1))
            plt.imshow(x_10[i,:].reshape(32,32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 10 PCs ',y=1)
        ims.tight_layout(pad=0,w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[1] + "/Q1|PART:B_pc_10.png", bbox_inches='tight')
        plt.show()


        # Plotting k = 25 Reconstructed 36 Images
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row,col,i+1))
            plt.imshow(x_25[i,:].reshape(32,32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 25 PCs ',y=1)
        ims.tight_layout(pad=0,w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[1] + "/Q1|PART:B_pc_25.png", bbox_inches='tight')
        plt.show()


        # Plotting k = 50 Reconstructed 36 Images
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row,col,i+1))
            plt.imshow(x_50[i,:].reshape(32,32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 50 PCs', y=1)
        ims.tight_layout(pad=0,w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[1] + "/Q1|PART:B_pc_50.png", bbox_inches='tight')
        plt.show()

        # Obtaining MSEs & Stds for 3 cases
        mse_10 = np.mean(np.square(faces-x_10))
        std_10 = np.std(np.mean(np.square(faces-x_10),axis=1))
        print('10 PCs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mse_10) + ' || Std = ' + str(std_10))
        mse_25 = np.mean(np.square(faces - x_25))
        std_25 = np.std(np.mean(np.square(faces-x_25),axis=1))
        print('25 PCs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mse_25) + ' || Std = ' + str(std_25))
        mse_50 = np.mean(np.square(faces - x_50))
        std_50 = np.std(np.mean(np.square(faces-x_50),axis=1))
        print('50 PCs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mse_50) + ' || Std = ' + str(std_50))





        print("Question 1: Part C")
        # Implementing ICA for the reduced 50-dimensional data
        # z_50 variable is the (1000x50) shaped PCA-reduced data

        # Obtaining ICA components in order to display them as images
        ica_10 = FastICA(10, random_state=np.random.seed(12)).fit(norm_faces)
        ica_25 = FastICA(25, random_state=np.random.seed(12)).fit(norm_faces)
        ica_50 = FastICA(50, random_state=np.random.seed(12)).fit(norm_faces)

        # Displaying respective outputs as images
        # 10 components case

        # Displaying 10 ICs as images
        ims = plt.figure()
        row = 2
        col = 5
        ax = []
        for i in range(10):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(ica_10.components_[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('10 Independent Components ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C_IC_10.png", bbox_inches='tight')
        plt.show()



        # Displaying 25 ICs as images
        ims = plt.figure()
        row = 5
        col = 5
        ax = []
        for i in range(25):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(ica_25.components_[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('25 Independent Components ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C_IC_25.png", bbox_inches='tight')
        plt.show()



        # Displaying 50 ICs as images
        ims = plt.figure()
        row = 5
        col = 10
        ax = []
        for i in range(50):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(ica_50.components_[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('50 Independent Components ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C_IC_50.png", bbox_inches='tight')
        plt.show()


        # Repeating the ICA procedure for PCA-reduced data
        # Implementing ICA for the reduced 50-dimensional data
        # z_50 variable is the (1000x50) shaped PCA-reduced data
        ica_10 = FastICA(10, random_state=np.random.seed(12)).fit(x_50)
        ica_25 = FastICA(25, random_state=np.random.seed(12)).fit(x_50)
        ica_50 = FastICA(50, random_state=np.random.seed(12)).fit(x_50)

        # Obtaining the respective Mixing Matrices -A

        A_10 = FastICA(10, random_state=np.random.seed(12)).fit(x_50).mixing_  # (50, 10)
        A_25 = FastICA(25, random_state=np.random.seed(12)).fit(x_50).mixing_  # (50, 25)
        A_50 = FastICA(50, random_state=np.random.seed(12)).fit(x_50).mixing_  # (50, 50)

        # Reconstructing the data matrix from 3 projection cases
        icrec_10 = np.dot(ica_10.transform(x_50), A_10.T) + ica_10.mean_ #(1000,50)
        icrec_25 = np.dot(ica_25.transform(x_50), A_25.T) + ica_25.mean_
        icrec_50 = np.dot(ica_50.transform(x_50), A_50.T) + ica_50.mean_

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(icrec_10[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 10 ICs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C|IC-rec1.png", bbox_inches='tight')
        plt.show()

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(icrec_25[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 25 ICs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C|IC-rec2.png", bbox_inches='tight')
        plt.show()

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(icrec_50[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 50 ICs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C|IC-rec3.png", bbox_inches='tight')
        plt.show()

        # Obtaining MSEs & Stds for 3 cases
        ic_mse_10 = np.mean(np.square(faces-icrec_10))
        ic_std_10 = np.std(np.mean(np.square(faces-icrec_10),axis=1))
        print('10 ICs Reconstruction MSE --> Mean and Std:  Mean = ' + str(ic_mse_10) + ' || Std = ' + str(ic_std_10))
        ic_mse_25 = np.mean(np.square(faces - icrec_25))
        ic_std_25 = np.std(np.mean(np.square(faces-icrec_25),axis=1))
        print('25 ICs Reconstruction MSE --> Mean and Std:  Mean = ' + str(ic_mse_25) + ' || Std = ' + str(ic_std_25))
        ic_mse_50 = np.mean(np.square(faces - icrec_50))
        ic_std_50 = np.std(np.mean(np.square(faces-icrec_50),axis=1))
        print('50 ICs Reconstruction MSE --> Mean and Std:  Mean = ' + str(ic_mse_50) + ' || Std = ' + str(ic_std_50))



        print("Question 1: Part D")

        # Obtaining the minimum pixel value within faces matrix
        min_pix = np.abs(np.min(faces))
        # Reinstating the non-negativite items constraint
        nnmf_data = faces + min_pix

        # Obtaining 3 cases of NMF procedure
        nnmf_10 = NMF(n_components=10, solver='mu', max_iter=600).fit(nnmf_data)
        nnmf_25 = NMF(n_components=25, solver='mu', max_iter=600).fit(nnmf_data)
        nnmf_50 = NMF(n_components=50, solver='mu', max_iter=600).fit(nnmf_data)

        # k = 10
        # Obtaining H expansion matrix component of shape (10,1024)
        h_10 = nnmf_10.components_
        # Obtaining W dictionary matrix component of shape (1000,10)
        w_10 = nnmf_10.transform(nnmf_data)

        # Displaying k=10 MFs

        ims = plt.figure()
        row = 2
        col = 5
        ax = []
        for i in range(10):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(h_10[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('10 MF Images ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_10.png", bbox_inches='tight')
        plt.show()


        # k = 25
        # Obtaining H expansion matrix component of shape (10,1024)
        h_25 = nnmf_25.components_
        # Obtaining W dictionary matrix component of shape (1000,10)
        w_25 = nnmf_25.transform(nnmf_data)
        # Displaying 25 MFs as images
        ims = plt.figure()
        row = 5
        col = 5
        ax = []
        for i in range(25):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(h_25[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('25 MF Images ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_25.png", bbox_inches='tight')
        plt.show()


        # k = 50
        # Obtaining H expansion matrix component of shape (10,1024)
        h_50 = nnmf_50.components_
        # Obtaining W dictionary matrix component of shape (1000,10)
        w_50 = nnmf_50.transform(nnmf_data)
        # Displaying 50 MFs as images
        ims = plt.figure()
        row = 5
        col = 10
        ax = []
        for i in range(50):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(h_50[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('50 MF Images', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_50.png", bbox_inches='tight')
        plt.show()

        # Reconstructing each Case
        rec_mf_10 = np.dot(w_10,h_10) - min_pix
        rec_mf_25 = np.dot(w_25, h_25) - min_pix
        rec_mf_50 = np.dot(w_50, h_50) - min_pix

        # Displaying the reconstructed images

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(rec_mf_10[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 10 MFs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_rec1.png", bbox_inches='tight')
        plt.show()

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(rec_mf_25[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 25 MFs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_rec2.png", bbox_inches='tight')
        plt.show()

        # Displaying the first 36 reconstructed images for 3 cases
        ims = plt.figure()
        row = 6
        col = 6
        ax = []
        for i in range(36):
            ax.append(ims.add_subplot(row, col, i + 1))
            plt.imshow(rec_mf_50[i, :].reshape(32, 32).T, cmap='gray')
            ax[-1].set_xticks([])
            ax[-1].set_yticks([])
            ax[-1].set_xlabel(i + 1)

        ims.suptitle('36 Images Reconstructed with 50 MFs ', y=1)
        ims.tight_layout(pad=0, w_pad=0, h_pad=0)

        #plt.savefig(q1_plot_directory[3] + "/Q1|PART:D_MF_rec3.png", bbox_inches='tight')
        plt.show()

        # Obtaining MSEs & Stds for 3 cases
        mf_mse_10 = np.mean(np.square(faces - rec_mf_10))
        mf_std_10 = np.std(np.mean(np.square(faces - rec_mf_10), axis=1))
        print('10 MFs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mf_mse_10) + ' || Std = ' + str(mf_std_10))
        mf_mse_25 = np.mean(np.square(faces - rec_mf_25))
        mf_std_25 = np.std(np.mean(np.square(faces - rec_mf_25), axis=1))
        print('25 MFs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mf_mse_25) + ' || Std = ' + str(mf_std_25))
        mf_mse_50 = np.mean(np.square(faces - rec_mf_50))
        mf_std_50 = np.std(np.mean(np.square(faces - rec_mf_50), axis=1))
        print('50 MFs Reconstruction MSE --> Mean and Std:  Mean = ' + str(mf_mse_50) + ' || Std = ' + str(mf_std_50))







    elif question == "2":
        np.random.seed(12)
        q2_plot_directory = ['Q2|A', 'Q2|B', 'Q2|C', 'Q2|D', 'Q2|E']


        print("Question 2")


        print("Question 2: Part A")

        # Defining the mean interval
        mean_rng = np.arange(-10,11,1)
        stims = np.linspace(-20, 21, 1000)

        # Plotting all tuning curves
        plt.figure()
        for m in mean_rng:
            resp = tuning_c(stims, 1, m)
            plt.plot(stims, resp)

        plt.title('All Tuning Curves on the Same Axis')
        plt.xlabel('Stimulus')
        plt.ylabel('Neural Response')
        #plt.savefig(q2_plot_directory[0] + "/Q2|PART:A|all.png", bbox_inches='tight')
        plt.show()

        # Simulating x = -1 response
        sim_response = tuning_c(-1,1,mean_rng)
        # Plotting the output
        plt.figure()
        plt.plot(mean_rng,sim_response)
        plt.title('Population Response to -1 Stimulus')
        plt.xlabel('Neurons Preferred Stimulus Value')
        plt.ylabel('Neural Response')
        #plt.savefig(q2_plot_directory[0] + "/Q2|PART:A|stim.png", bbox_inches='tight')
        plt.show()




        print("Question 2: Part B")

        # Defining the stimulus interval
        stim_rng = np.linspace(-5, 5, 100)

        act_stims = list()
        resp_list = list()
        x_wta = list()
        est_error = list()
        # Conducting the experiment
        for i in range(200):

            # Sampling stimulus from its interval
            stim_samp = np.random.choice(stim_rng)
            act_stims.append(stim_samp)
            # Response without noise
            pure_resp = tuning_c(stim_samp, 1, mean_rng)
            # Additive Gaussian Noise
            agn = np.random.normal(0, 0.05, 21)
            # Corrupting the response
            real_resp = pure_resp + agn
            resp_list.append(real_resp)

            # Winner Takes All Estimate
            winning = winner_takes_all(mean_rng,real_resp)
            x_wta.append(winning)

            # Estimation Error
            err = np.abs(winning-stim_samp)
            est_error.append(err)


        # Plotting actual and estimated stimulus
        trials = np.arange(0, 200)
        plt.figure()
        # Plotting actual stimulus
        plt.scatter(trials, act_stims)
        # Plotting estimated stimulus
        plt.scatter(trials, x_wta)
        plt.title('Actual and WTA-Estimated Stimuli')
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.legend(['Actual Stimulus', 'WTA Estimate'], loc='best')
        #plt.savefig(q2_plot_directory[1] + "/Q2|PART:B|scat.png", bbox_inches='tight')
        plt.show()

        # Calculating the mean and std of estimation error
        est_error = np.array(est_error)
        mean_error = np.mean(est_error)
        std_error = np.std(est_error)
        print('WTA Estimation Error --> Mean and Std:  Mean = ' + str(mean_error) + ' || Std = ' + str(std_error))


        print("Question 2: Part C")

        # ML decoder estimation loop
        x_mle = list()
        error_mle = list()
        for k in range(200):
            act_stim = act_stims[k]
            ml_est = ML_decoder(mean_rng, stim_rng, resp_list[k])
            x_mle.append(ml_est)
            mle_error = np.abs(act_stim - ml_est)
            error_mle.append(mle_error)

        # Plotting actual and estimated stimulus
        plt.figure()
        # Plotting actual stimulus
        plt.scatter(trials, act_stims)
        # Plotting estimated stimulus with ML-decoder
        plt.scatter(trials, x_mle)
        plt.title('Actual and ML-Estimated Stimuli')
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.legend(['Actual Stimulus', 'ML Estimate'], loc='best')
        #plt.savefig(q2_plot_directory[2] + "/Q2|PART:C|scat.png", bbox_inches='tight')
        plt.show()

        # Calculating the mean and std of estimation error
        error_mle = np.array(error_mle)
        mean_ml_error = np.mean(error_mle)
        std_ml = np.std(error_mle)
        print('MLE Estimation Error --> Mean and Std:  Mean = ' + str(mean_ml_error) + ' || Std = ' + str(std_ml))



        print("Question 2: Part D")

        # MAP Decoder estimation loop
        x_map = list()
        error_map = list()
        for k in range(200):
            act_stim = act_stims[k]
            map_est = MAP_decoder(mean_rng, stim_rng, resp_list[k])
            x_map.append(map_est)
            map_error = np.abs(act_stim - map_est)
            error_map.append(map_error)

        # Plotting actual and estimated stimulus
        plt.figure()
        # Plotting actual stimulus
        plt.scatter(trials, act_stims)
        # Plotting estimated stimulus with MAP-decoder
        plt.scatter(trials, x_map)
        plt.title('Actual and MAP-Estimated Stimuli')
        plt.xlabel('Trials')
        plt.ylabel('Stimuli')
        plt.legend(['Actual Stimulus', 'MAP Estimate'], loc='best')
        #plt.savefig(q2_plot_directory[3] + "/Q2|PART:D|scat.png", bbox_inches='tight')
        plt.show()

        # Calculating the mean and std of estimation error
        error_map = np.array(error_map)
        mean_map_error = np.mean(error_map)
        std_map = np.std(error_map)
        print('MAP Estimation Error --> Mean and Std:  Mean = ' + str(mean_map_error) + ' || Std = ' + str(std_map))

        print("Question 2: Part E")

        actual_stims = list()
        # Generating the stimulus samples for the new experiment
        for k in range(200):
            stim_samples = np.random.choice(stim_rng)
            actual_stims.append(stim_samples)


        # Running the experiment for 6 different std parameter cases
        std_cases = [0.1, 0.2, 0.5, 1, 2, 5]
        error_means = list()
        error_stds = list()

        case = 0
        for std_i in std_cases:
            case +=1
            mle_mean, mle_std = sim_experiment(actual_stims, std_i, mean_rng)
            error_means.append(mle_mean)
            error_stds.append(mle_std)
            # Displaying the obtained Error Metrics
            print('For Sigma = ' + str(std_i) + ' :')
            print('MLE Error --> Mean and Std:  Mean = ' + str(mle_mean) + ' || Std = ' + str(mle_std))

            if case < 6:
                print('------------------------------------------------------')








# Functions used in the assignment in order

# Functions used in Question 2

# Tuning Curve Function
def tuning_c(stim, std, mean):

    f = np.exp((-1*np.square(stim-mean))/(2* (std**2)))
    return f

# Winner Takes All
def winner_takes_all(chosen_one, resp):
    winning_number = np.argmax(resp)
    winner = chosen_one[winning_number]
    return winner

# Maximum Likelihood Decoder
def ML_decoder(mean_interv, stim_interv, resps):
    nlog_l = list()

    for stim in stim_interv:
        nlog_sum = 0
        for r_out, mean in zip(resps, mean_interv):
            nlog_sum += np.square(r_out - tuning_c(stim, 1, mean))

        nlog_l.append(nlog_sum)

    nlog = np.array(nlog_l)
    min_neuron = np.argmin(nlog)
    ml_est = stim_interv[min_neuron]

    return ml_est


# MAP Decoder
def MAP_decoder(mean_interv, stim_interv, resps):
    cons_term = 200
    posterior_list = list()

    for stim in stim_interv:

        # prior = stim^2/(2*(2.5)^2) = stim^2 / 12.5
        nlog_pr = np.square(stim) / (2 * np.square(2.5))

        nlog_posterior = 0
        nlog_lhood = 0

        for r_out, mean in zip(resps, mean_interv):
            nlog_lhood += np.square(r_out - tuning_c(stim, 1, mean))

        nlog_posterior = (nlog_lhood * cons_term) + nlog_pr

        posterior_list.append(nlog_posterior)

    map_arr = np.array(posterior_list)
    min_map = np.argmin(map_arr)
    map_est = stim_interv[min_map]

    return map_est

# Maximum Likelihood Decoder for part E
def ML_decoder_E(mean_interv, stim_interv, resps, std_i):
    nlog_l = list()

    for stim in stim_interv:
        nlog_sum = 0
        for r_out, mean in zip(resps, mean_interv):
            nlog_sum += np.square(r_out - tuning_c(stim, std_i, mean))

        nlog_l.append(nlog_sum)

    nlog = np.array(nlog_l)
    min_neuron = np.argmin(nlog)
    ml_est = stim_interv[min_neuron]

    return ml_est

# Overall Simulation Function

def sim_experiment(act_stims, std_val, mean_interv):

    np.random.seed(12)
    s_range = np.linspace(-5, 5, 100)
    est_error = list()
    # Conducting the experiment for input sigma parameter
    for i in range(200):
        # Sampled stimulus
        stim_samp = act_stims[i]
        # Response without noise
        pure_resp = tuning_c(stim_samp, std_val, mean_interv)
        # Additive Gaussian Noise
        agn = np.random.normal(0, 0.05, 21)
        # Corrupting the response
        real_resp = pure_resp + agn

        # MLE Estimate
        ml_estimate = ML_decoder_E(mean_interv, s_range, real_resp,std_val)

        # Estimation Error
        mle_error = np.abs(stim_samp - ml_estimate)
        est_error.append(mle_error)


    # Obtaining Error Metrics
    # Calculating the mean and std of estimation error
    mle_error = np.array(est_error)
    ml_mean = np.mean(mle_error)
    ml_std = np.std(mle_error)

    return ml_mean,ml_std



arman_budunoglu_21602635_hw4(question)



