import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
import h5py
import scipy.stats



question = input("Enter Relevant Question Number to Display its Respective Output \n")

def arman_budunoglu_21602635_hw3(question):

    if question == "1":

        q1_plot_directory = ['Q1|A', 'Q1|B', 'Q1|C']


        print("Question 1: ")

        # Loading the dataset
        hf = h5py.File('hw3_data2.mat', 'r')
        ls = list(hf.keys())
        # Examining its contents
        print("Contents of hw3_data2.mat file: " + str(ls))


        print("Question 1: Part A")
        # Inspecting Yn array
        Y_n = np.array(hf.get('Yn'))
        print('Yn is of shape: ' + str(Y_n.shape)) # (1, 1000)
        # Inspecting Xn
        X_n = np.array(hf.get('Xn'))
        print('Xn is of shape: ' + str(X_n.shape))  # (100, 1000)

        # Reshaping Response in vector format
        Y_response = Y_n.T  # (1000,1)
        dataset = X_n.T  # (1000,100)
        y_res = Y_response
        test_metric = list()
        val_metric = list()
        # generating the lambda interval
        penalty_interv = np.logspace(0, 12, num=100, base=10)

        # training and cross validation loop

        for lmb in penalty_interv:
            cv_val = list()
            cv_test = list()
            # performing over k-folds
            for k in range(0, 10):

                y_arr = y_res
                x_arr = dataset
                if k == 9:

                    val_data = dataset[k * 100: k * 100 + 100]
                    val_y = y_res[k * 100: k * 100 + 100]
                    test_data = dataset[0:100]
                    test_y = y_res[0:100]
                else:
                    val_data = dataset[k * 100: k * 100 + 100]
                    val_y = y_res[k * 100: k * 100 + 100]
                    test_data = dataset[k * 100 + 100:(k + 2) * 100]
                    test_y = y_res[k * 100 + 100:(k + 2) * 100]

                # Splitting the training data with the test/val instances removed
                train_data, train_y = train_split(k, x_arr, y_arr)

                # obtaining the B_ridge on training data
                B_rdg = Ridge_Reg(train_y, train_data, lmb)  # (100,1)

                # obtaining R square for each fold in both test/val instances
                R2_test = kfold_metrics(test_y, B_rdg, test_data)
                R2_val = kfold_metrics(val_y, B_rdg, val_data)
                cv_val.append(R2_val)
                cv_test.append(R2_test)

            avrg_val = np.mean(np.array(cv_val))
            avrg_test = np.mean(np.array(cv_test))

            # appending the mean test/val error metrics
            val_metric.append(avrg_val)
            test_metric.append(avrg_test)


        # Turning error lists into Array's for finding the max R^2 index
        val_metric = np.array(val_metric)
        test_metric = np.array(test_metric)

        # Finding the best performing lambda value
        max_index = np.argmax(val_metric)
        opt_lmbd = penalty_interv[max_index]
        r2_max = val_metric[max_index]
        print('Optimal Value of Lambda is: ' + str(opt_lmbd))
        print('Corresponding R^2 at Validation: ' + str(r2_max))



        # Plotting R^2 Curves for test/val together

        plt.figure()
        # Validation R^2 vs Lambda
        plt.plot(penalty_interv,val_metric)
        # Test R^2 vs Lambda
        plt.plot(penalty_interv,test_metric)
        plt.xscale('log')
        plt.xlabel('Penalty Term')
        plt.ylabel('R^2')
        plt.title('Validation/Test R^2 vs Lambda')
        plt.legend(['Validation', 'Test'])
        #plt.savefig(q1_plot_directory[0] + "/Q1|PART:A.png", bbox_inches='tight')
        plt.show()




        print("Question 1: Part B")
        # plt.savefig(q2_plot_directory[0] + "/Q2|PART:A.png", bbox_inches='tight')

        # Creating the Bootstrap sample coefficients loop
        np.random.seed(12)
        coeff_list = list()
        boot = 500
        x_regs = X_n.T
        y_resps = Y_n.T
        for b in range(boot):
            #  Generating random sample indices with replacement
            samp_index = np.random.randint(1000, size=1000)
            synth_samp = x_regs[samp_index]
            y_samp = y_resps[samp_index]
            # Obtaining coefficients for the bootstrap sample with lmbd = 0 OLS soln
            B_sample = Ridge_Reg(y_samp, synth_samp, 0)
            coeff_list.append(B_sample)


        # Turning the obtained 500 coefficient samples into an array for mean and
        # 95% confidence interval calculations
        boot_w = np.array(coeff_list).reshape(500,100)
        # Obtaining mean
        coeff_mean = np.mean(boot_w,axis=0)
        # 95% confidence intervals at +- 2std 's from the center
        coeff_std = np.std(boot_w,axis=0)

        # Obtaining weights that are significantly different than 0
        sig_w_ind = sig_weights(coeff_mean,coeff_std)[0]
        print('Significant Weight Indices: ' + str(sig_w_ind))


        # Generating the array containing significant weights means
        sig_means = np.ones_like(coeff_mean)
        # the remainder non-significant indices out of 100 are given 1 value for labelling
        # only necessary indices in overall plot
        sig_means[sig_w_ind] = coeff_mean[sig_w_ind]
        # Plotting mean and conf intervals along with means of significant weight indices
        plt.figure()
        plt.figure(figsize=(15, 10))
        # Setting x axis as regressor number range
        reg_range = np.arange(0,100)
        conf_interv = 2 * coeff_std
        # Plotting CI
        plt.errorbar(reg_range, coeff_mean, yerr=conf_interv, ecolor='r', fmt='none', capsize=5)
        # Significant index indicator as means of those indices
        plt.plot(sig_means, 'x', color='b', markersize=15)
        # Overall Means
        plt.plot(coeff_mean, 'o', color='black')
        plt.title('OLS Model Parameters Mean and CI')
        plt.xlabel('Weight Index: ' r'$\beta_i$')
        plt.ylabel('Weight Values')
        plt.legend(['Significant Weight Mean','Mean','95% CI'])
        plt.ylim((-0.18566343526569992, 0.3161325177809522))
        #plt.savefig(q1_plot_directory[1] + "/Q1|PART:B.png", bbox_inches='tight')
        plt.show()






        print("Question 1: Part C")

        # Repeating the bootstrapping process in part B for
        # Ridge Reg Solution weights with optimal lambda found in part A
        # Creating the Bootstrap sample coefficients loop
        Rdg_coeffs = list()
        boot = 500
        for b in range(boot):
            #  Generating random sample indices with replacement
            samp_index = np.random.randint(1000, size=1000)
            synth_samp = x_regs[samp_index]
            y_samp = y_resps[samp_index]
            # Obtaining coefficients for the bootstrap sample with lmbd = 0 OLS soln
            B_sample = Ridge_Reg(y_samp, synth_samp, opt_lmbd)
            Rdg_coeffs.append(B_sample)

        #  mean and 95% confidence interval calculations for Ridge Solution
        boot_rdg = np.array(Rdg_coeffs).reshape(500, 100)
        # Obtaining mean
        rdg_mean = np.mean(boot_rdg, axis=0)
        # 95% confidence intervals at +- 2std 's from the center
        rdg_std = np.std(boot_rdg, axis=0)

        # Obtaining weights that are significantly different than 0
        sig_rdg_ind = sig_weights(rdg_mean, rdg_std)[0]
        print('Significant Weight Indices: ' + str(sig_rdg_ind))

        # Repeating the same procedure for plotting in part B
        # Generating the array containing significant weights means

        sig_means_rdg = np.ones_like(rdg_mean)
        # the remainder non-significant indices out of 100 are given 1 value for labelling
        # only necessary indices in overall plot
        sig_means_rdg[sig_rdg_ind] = rdg_mean[sig_rdg_ind]
        # Plotting mean and conf intervals along with means of significant weight indices
        plt.figure()
        plt.figure(figsize=(15, 10))

        conf_rdg = 2 * rdg_std
        # Plotting CI
        plt.errorbar(reg_range, rdg_mean, yerr=conf_rdg, ecolor='r', fmt='none', capsize=5)
        # Significant index indicator as means of those indices
        plt.plot(sig_means_rdg, 'x', color='b', markersize=15)
        # Overall Means
        plt.plot(rdg_mean, 'o', color='black')
        plt.title('Ridge Model Parameters Mean and CI')
        plt.xlabel('Weight Index: ' r'$\beta_i$')
        plt.ylabel('Weight Values')
        plt.legend(['Significant Weight Mean', 'Mean', '95% CI'])
        plt.ylim((-0.1, 0.15))
        #plt.savefig(q1_plot_directory[2] + "/Q1|PART:C.png", bbox_inches='tight')
        plt.show()






    elif question == "2":
        q2_plot_directory = ['Q2|A', 'Q2|B', 'Q2|C', 'Q2|D', 'Q2|E']

        print("Question 2")


        # Loading the dataset for this question
        hf = h5py.File('hw3_data3.mat', 'r')
        ls = list(hf.keys())
        # Examining its contents
        print("Contents of hw3_data3.mat file: " + str(ls))


        print("Question 2: Part A")

        np.random.seed(12)
        # Getting pop1 pop2 variables
        pop_1 = np.array(hf.get('pop1'))
        pop_2 = np.array(hf.get('pop2'))
        print('pop1 shape:' + str(pop_1.shape)) # (7, 1)
        print('pop2 shape:' + str(pop_2.shape)) # (5, 1)

        # Merging two pop arrays in order to conduct null hypothesis test
        pops = np.concatenate((pop_1,pop_2))
        # bootstrapping 10000 iterations on merged samples
        boot_pops = bootstrap_samples(pops) # 10000,12
        # Seperating the merged bootstrapped samples
        pb_1, pb_2 = np.hsplit(boot_pops, [7])
        # Calculating the respective means
        pb1_mean = np.mean(pb_1, axis=1)
        pb2_mean = np.mean(pb_2, axis=1)

        # Equating the difference of means
        mean_diff = pb1_mean - pb2_mean
        #Plotting it as a histogram
        binwidth = 0.1
        plt.figure()
        plt.hist(mean_diff,bins=np.arange(min(mean_diff),max(mean_diff) + binwidth, binwidth), edgecolor='black', density=True)
        plt.ylabel('P(x|H_0)')
        plt.xlabel('Mean Difference|H_0')
        plt.title('Bootstrapped Samples Population Mean Difference')
        #plt.savefig(q2_plot_directory[0] + "/Q2|PART:A.png", bbox_inches='tight')
        plt.show()

        # Equating two Tailed p-value
        m_1 = np.mean(pop_1)
        m_2 = np.mean(pop_2)
        m_d = m_1 - m_2
        p_a = p_value(mean_diff,m_d)
        print('Two Tailed p-value : ' + str(2*p_a))



        print("Question 2: Part B")

        # Obtaining BOLD responses in vox arrays
        vox_1 = np.array(hf.get('vox1'))
        vox_2 = np.array(hf.get('vox2'))
        print('vox1 shape:' + str(vox_1.shape)) # (50, 1)
        print('vox2 shape:' + str(vox_2.shape)) # (50, 1)

        # Identically Bootstrapping vox arrays
        boot_vox1,boot_vox2 = bootstrap_identical(vox_1,vox_2)


        # Obtaining their correlation
        # initializing a list to store correlations
        vox_corr = list()
        for i in range(10000):
            sample_cor = np.corrcoef(boot_vox1[i], boot_vox2[i])[0, 1]
            cor_val = sample_cor
            vox_corr.append(cor_val)

        vox_corr = np.array(vox_corr)
        # mean of the correlation
        vox_c_mean = np.mean(vox_corr)
        print('Mean of the correlation = ' + str(vox_c_mean))
        # Computing 95% CI
        # Corresponding CI lower upper bounds within 10000 data points
        conf_interval = np.percentile(vox_corr, [2.5, 97.5])
        print('95% CI lower and upper bounds -->: '+ str(conf_interval))
        # 0 correlation indices
        zero_corr_indices = np.where(vox_corr == 0)
        print('There are no elements with 0 correlation')
        print('Element indices with 0 correlation: ' + str(zero_corr_indices))









        print("Question 2: Part C")

        # Independently bootstrapping vox1 vox2 responses
        boot_v1 = bootstrap_samples(vox_1)
        boot_v2 = bootstrap_samples(vox_2)

        # Obtaining their correlation
        # initializing a list to store correlations
        indep_corr = list()
        for i in range(10000):
            sample_cor = np.corrcoef(boot_v1[i], boot_v2[i])[0, 1]
            indep_corr.append(sample_cor)

        indep_corr = np.array(indep_corr)


        # plotting bootstrapped correlations as a histogram
        # Normalizing the histogram
        norm = np.ones_like(indep_corr)/float(len(indep_corr))
        binwidth = 0.02
        plt.figure()
        plt.hist(indep_corr, bins=np.arange(min(indep_corr), max(indep_corr) + binwidth, binwidth), edgecolor='black',
                  weights=norm)
        plt.ylabel('P(x|H_0)')
        plt.xlabel('Correlation|H_0')
        plt.title('Independently Bootstrapped Samples Correlation')
        #plt.savefig(q2_plot_directory[2] + "/Q2|PART:C.png", bbox_inches='tight')
        plt.show()

        # mean of the bootstrapped correlations
        ind_cor_mean = np.mean(indep_corr)
        print('Independent Sampling | Mean of the correlation = ' + str(ind_cor_mean))
        # obtaining original data correlations
        corr_og = np.corrcoef(vox_1.flatten(), vox_2.flatten())[0, 1]
        # Calculating the one Tailed p-value
        one_Tailed_p = p_value(indep_corr,corr_og)
        print('One Tailed p-value : ' + str(one_Tailed_p))






        print("Question 2: Part D")

        # loading the building and face image responses
        resp_face = np.array(hf.get('face'))
        resp_build = np.array(hf.get('building'))

        print('face response shape:' + str(resp_face.shape)) # (20, 1)
        print('building response shape:' + str(resp_build.shape)) # (20, 1)

        # Same subject population bootstrapping algorithm for this question
        resp_face = resp_face.flatten()
        resp_build = resp_build.flatten()
        boot_subj_mean = list()
        possible_outcomes = np.zeros(4)
        for i in range(10000):
            subjects = list()
            for j in range(20):
                subject = np.random.randint(0, 20)
                possible_outcomes[2] = resp_build[subject] - resp_face[subject]
                possible_outcomes[3] = resp_face[subject] - resp_build[subject]
                sample_out = possible_outcomes[np.random.randint(0, 4)]
                subjects.append(sample_out)

            mean_val = np.mean(np.array(subjects))
            boot_subj_mean.append(mean_val)

        boot_subj_mean = np.array(boot_subj_mean)
        # Plotting the obtained mean array
        binsize = 60
        plt.figure()
        plt.hist(boot_subj_mean, bins=binsize, edgecolor='black',
                 weights=norm)
        plt.ylabel('P(x|H_0)')
        plt.xlabel('Mean Difference|H_0')
        plt.title('Bootstrapped Samples Subjects Mean Difference')
        #plt.savefig(q2_plot_directory[3] + "/Q2|PART:D.png", bbox_inches='tight')
        plt.show()

        # Calculating the p value
        mean_og = np.mean(resp_face) - np.mean(resp_build)
        # Calculating the one Tailed p-value
        two_Tailed_p = p_value(boot_subj_mean, mean_og)
        print('Two Tailed p-value : ' + str(two_Tailed_p*2))



        print("Question 2: Part E")

        # Reloading the dataset to fit the previously written functions
        resp_face = np.array(hf.get('face'))
        resp_build = np.array(hf.get('building'))
        # Merging two response arrays in order to conduct null hypothesis test
        resps = np.concatenate((resp_face, resp_build))
        # bootstrapping 10000 iterations on merged samples
        boot_resps = bootstrap_samples(resps)  # 10000,12
        # Seperating the merged bootstrapped samples
        r1, r2 = np.hsplit(boot_resps, [20])
        # Calculating the respective means
        r1_mean = np.mean(r1, axis=1)
        r2_mean = np.mean(r2, axis=1)

        # Equating the difference of means
        mean_diff_e = r1_mean - r2_mean
        # Equating two Tailed p-value
        m1 = np.mean(resp_face)
        m2 = np.mean(resp_build)
        m_e = m1 - m2
        p_val_e = p_value(mean_diff_e, m_e)
        print('Two Tailed p-value : ' + str(p_val_e * 2))

        # Plotting it as a histogram
        # Plotting it as a histogram
        plt.figure()
        plt.hist(mean_diff_e, bins=binsize, edgecolor='black', weights=norm)
        plt.ylabel('P(x|H_0)')
        plt.xlabel('Mean Difference|H_0')
        plt.title('Bootstrapped Distinct Subject Population Mean Difference')
        #plt.savefig(q2_plot_directory[4] + "/Q2|PART:E.png", bbox_inches='tight')
        plt.show()










# Functions used in the assignment in order
# R square variance proportion metric function
def R_2(actual,predict):
    # Numpy's corcoeff function calculates the pearson correlation coeff
    cor_coef = np.corrcoef(actual,predict)
    r = cor_coef[1,0]
    r_square = np.square(r)
    return r_square



# Ridge regression solution function
def Ridge_Reg(Y_resp,X,p):

    # Closed form algebraic solution of Ridge Reg Solution is as follows
    reg_row,reg_col = X.shape
    penalty = p * np.identity(reg_col)
    inv_matrix = np.linalg.inv(np.dot(X.T,X) + penalty) # (100,100)
    pre_cal = np.dot(inv_matrix,X.T) # (100,1000)
    B_ridge = np.dot(pre_cal,Y_resp) # (100,1) 

    return B_ridge

# Removing test/val instances from the overall dataset for train set generation
def train_split(index, x_data, y_data):
    all_x = x_data
    all_y = y_data
    if index == 9:
        train_data = all_x[200:900]
        train_y = all_y[200:900]

    else:
        all_x = x_data
        all_y = y_data
        start_index = index * 100
        stop_index = ((index + 2) * 100) - 1
        remove_range = np.linspace(start_index, stop_index, 200).astype(int)
        train_data = np.delete(all_x, remove_range, axis=0)
        train_y = np.delete(all_y, remove_range, axis=0)

    return train_data, train_y


# obtains prediction and R_2 metric
def kfold_metrics(act_y, B_coef, x_ins):
    pred = np.dot(x_ins,B_coef)
    cv_fold = R_2(act_y.T, pred.T)

    return cv_fold

# obtaining indexes of the weights significantly different than 0
def sig_weights(c_mean,c_std):
    sig = np.abs(c_mean/c_std)
    sig_level = 2 - (2 * scipy.stats.norm.cdf(sig))
    sig_index = np.where(sig_level < 0.05)

    return sig_index

# Functions used in Q2
def bootstrap_samples(samples):
    s_num = samples.shape[0]
    dataset = samples.flatten()
    synth_list = list()
    for b in range(10000):
        synth_sample = np.random.choice(dataset,s_num)
        synth_list.append(synth_sample)

    return np.array(synth_list)

# Function that computes the one Tailed p value
def p_value(dist,samp_mean_diff):
    z_val = (samp_mean_diff-np.mean(dist))/np.std(dist)
    p_val = 1 - scipy.stats.norm.cdf(z_val)

    return p_val

# Identical sampling bootstrap function
def bootstrap_identical(arr_1, arr_2):
    # For identical sampling we will sample by the indices
    data_1 = arr_1.flatten()
    data_2 = arr_2.flatten()
    s_num = arr_1.shape[0]
    b_1 = list()
    b_2 = list()
    for b in range(10000):
        samp_indices = np.random.randint(50,size=50)
        b_1.append(data_1[samp_indices])
        b_2.append(data_2[samp_indices])

    b_1 = np.array(b_1)
    b_2 = np.array(b_2)
    return b_1, b_2



arman_budunoglu_21602635_hw3(question)



