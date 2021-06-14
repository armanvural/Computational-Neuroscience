import numpy as np
import matplotlib.pyplot as plt


question = input("Enter Relevant Question Number to Display its Respective Output \n")

def arman_budunoglu_21602635_hw1(question):
    if question == "1":

        print("Question 1: ")
        # Defining Ax = b system of equations
        # Transfer Function "A"
        A = np.array([[1,0,-1,2], [2,1,-1,5], [3,3,0,9]])  # 3x4 matrix
        # Output Vector "b"
        b = np.array([[1],[4],[9]])  # 3x1 vector

        print("Question 1: Part A")
        np.random.seed(35)
        # Randomly assigned free variables x_3 and x_4
        x_3 = np.random.rand()
        x_4 = np.random.rand()
        # Defining hand derived general solution x_n
        x_n = (x_3 * np.array([[1], [-1], [1], [0]])) + (x_4 * np.array([[-2], [-1], [0], [1]]))
        # Checking if the system Ax = 0 holds
        a_out = np.dot(A,x_n)
        print("A.x_n = \n", a_out)


        print("Question 1: Part B")
        # For x_3 = 2 and x_4 = 0
        # Plugging hand derived particular solution x_p
        x_p = np.array([[3], [0], [2], [0]])
        # Checking if b_out = [1 4 9]^T holds
        b_out = np.dot(A,x_p)
        print("A.x_p = \n",b_out)

        print("Question 1: Part C")
        # Using the randomly initialized free variables in part A
        # Defining the hand derived solution for part C
        x_c = (np.array([[1], [2], [0], [0]]) + (x_3 * np.array([[1], [-1], [1], [0]])) + (x_4 * np.array([[-2], [-1], [0], [1]])))
        # Checking if c_out = [1 4 9]^T holds
        c_out = np.dot(A,x_c)
        print("A.x_c = ")
        print(c_out)

        print("Question 1: Part D")
        # A_pseudo = Vx S_pseudo x U.T
        # Finding left singular eigenvectors for U
        left = np.dot(A,A.T)
        # Obtaining the normalized eigenvalues and their corresponding eigenvectors for AxA.T
        eig_u, eigvect_u = np.linalg.eigh(left)
        # Ordering the eigenvalue array
        eig_val_0 = eig_u[0]
        eig_u[0] = eig_u[2]
        eig_u[2] = eig_val_0
        # Ordering and reshaping the eigenvectors from linalg function in order to generate U
        u_1 = eigvect_u[:, 2].reshape(3,1)
        u_2 = eigvect_u[:, 1].reshape(3,1)
        u_3 = eigvect_u[:, 0].reshape(3,1)
        # Generating the left singular U matrix
        U_hand = np.concatenate([u_1, u_2, u_3], axis=1)
        # Generating the Sigma matrix
        S_hand = np.zeros((3,4))
        eig_to_sig = np.sqrt(eig_u.reshape(3,1))

        for i in range(3):
            S_hand[i][i] = np.around(eig_to_sig[i][0], 6)
        # Checking with linalg.svd function
        U_svd, S_svd, Vt_svd = np.linalg.svd(A)
        # Checking whether hand derived results match the svd function for U
        print("Hand Derivation for U = \n", U_hand)
        print("**********************")
        print("SVD function for U = \n", U_svd)

        if (U_hand.all() == U_svd.all()).all():
            print("Hand Derivation Algorithm matched SVD function for U")
        # Checking if they match for Sigma
        # Generating the Sigma matrix from SVD function
        S_func = np.zeros((3,4))
        for i in range(3):
            S_func[i][i] = np.around(S_svd[i], 6)
        S_svd = S_func

        print("Hand Derivation for Sigma = \n", S_hand)
        print("**********************")
        print("SVD function for Sigma = \n", S_svd)
        if (S_hand == S_svd).all():
            print("Hand Derivation Algorithm matched SVD function for Sigma")

        # Right singular matrix V
        print("SVD function for V.T = \n", Vt_svd)
        print("**********************")
        # For the pseudoinverse of A we need the inverse of sigma however it's not a square matrix
        # So we obtain sigma's pseudo inverse by obtaining (1/Sigma).T matrix
        for i in range(2):
            S_svd[i][i] = 1/S_svd[i][i]

        Sigma_pseudo = S_svd.T
        print("Pseudo Inverse of Sigma = \n", Sigma_pseudo)
        print("**********************")

        # Computing hand derived pseudo-inverse for A
        A_pseudo = np.dot(np.dot(Vt_svd.T,Sigma_pseudo),U_svd.T)
        print("Pseudo Inverse of A from Hand Derivation Algorithm is = ")
        print(A_pseudo)
        print("********************")
        # Direct Computer Verification
        pseudo_comp = np.linalg.pinv(A)
        print("Pseudo Inverse of A from pinv function is = ")
        print(pseudo_comp)
        if (A_pseudo.all() == pseudo_comp.all()).all():
            print("Pseudo Inverse is Correct")

        print("Question 1: Part E")
        # sparsest solution
        x_e = np.array([[1], [2], [0], [0]])
        e_out = np.dot(A,x_e)
        print("Sparsest solution with 2 non zero entries = \n ", e_out)
        print("Question 1: Part F")
        x_ln = np.dot(A_pseudo,b)
        print("Least-Norm Solution = \n ",x_ln)


    elif question == "2":
        print("Question 2")
        # Defining the examination results
        lang_tot = 869
        lang_active = 103
        nlang_tot = 2353
        nlang_active = 199

        print("Question 2: Part A")
        # Generating the x interval
        x_range = np.arange(0,1.001,0.001)

        # Finding the likelihood of language condition
        L_l = likelihood_a(lang_tot,lang_active,x_range)
        # Plotting it as a bar graph
        plt.figure()
        x_plot = np.arange(1001)
        plt.bar(x_plot, L_l, color='k')
        plt.title("Likelihood of Language Task")
        plt.xlabel("Probability x 1000")
        plt.ylabel("Likelihood")
        # Probabilities higher than 0.18 are approximately 0 thus
        plt.xlim(0, 180)
        #plt.savefig(" Q2|PART:A: Lang_Likelihood.png",bbox_inches='tight')
        plt.show()

        # Finding the likelihood of No-language condition
        L_nl = likelihood_a(nlang_tot,nlang_active,x_range)
        # Plotting it as a bar graph
        plt.figure()
        plt.bar(x_plot, L_nl, color='r')
        plt.title("Likelihood of Non-Language Task")
        plt.xlabel("Probability x 1000")
        plt.ylabel("Likelihood")
        # Probabilities higher than 0.18 are approximately 0 thus
        plt.xlim(0, 180)
        #plt.savefig(" Q2|PART:A: No-Lang_Likelihood.png",bbox_inches='tight')
        plt.show()

        print("Question 2: Part B")
        # Index of maximum x_l value
        max_xl = np.amax(L_l)
        index_max_l = np.argmax(L_l)/1000
        print("Maximum Likelihood for X_L is = " + str(max_xl))
        print("Probability that maximizes X_L is = " + str(index_max_l))
        # Index of maximum x_nl value
        max_xnl = np.amax(L_nl)
        index_max_nl = np.argmax(L_nl)/1000
        print("Maximum Likelihood for X_NL is = " + str(max_xnl))
        print("Probability that maximizes X_NL is = " + str(index_max_nl))

        print("Question 2: Part C")
        # Defining the uniformly distributed P(X)
        P_X = 1/1001
        # Posterior PDF's
        posterior_Xl = (L_l*P_X)/(np.sum(L_l*P_X))
        posterior_Xnl = (L_nl*P_X)/(np.sum(L_nl*P_X))
        # Plotting the posterior PDF's
        # Language Posterior Plot
        plt.figure()
        plt.bar(x_plot, posterior_Xl, color='k')
        plt.title("Posterior PDF of Language")
        plt.xlabel("Probability x 1000")
        plt.ylabel("Posterior Probability P(X_l|A)")
        # Probabilities higher than 0.18 are approximately 0 thus
        plt.xlim(0, 180)
        #plt.savefig(" Q2|PART:C: Lang_Post.png",bbox_inches='tight')
        plt.show()
        # No-Language Plot
        plt.figure()
        plt.bar(x_plot, posterior_Xnl, color='r')
        plt.title("Posterior PDF of No-Language")
        plt.xlabel("Probability x 1000")
        plt.ylabel("Posterior Probability P(X_nl|A)")
        # Probabilities higher than 0.18 are approximately 0 thus
        plt.xlim(0, 180)
        #plt.savefig(" Q2|PART:C: NOLang_Post.png",bbox_inches='tight')
        plt.show()

        # Deriving the CDFs for both cases
        posterior_cdf_l = np.zeros(1001)
        posterior_cdf_nl = np.zeros(1001)
        # Cumulatively adding probabilites for obtaining cdf's
        cumulative_prev_l = 0
        cumulative_prev_nl = 0
        for i in range(1001):
            posterior_cdf_l[i] += posterior_Xl[i] + cumulative_prev_l
            cumulative_prev_l = posterior_cdf_l[i]

            posterior_cdf_nl[i] += posterior_Xnl[i] + cumulative_prev_nl
            cumulative_prev_nl = posterior_cdf_nl[i]

        # Plotting the CDFs
        # Language CDF
        plt.figure()
        plt.bar(x_plot, posterior_cdf_l, color='k')
        plt.title("Posterior CDF of Language")
        plt.xlabel("x_l * 1000")
        plt.ylabel("Posterior CDF Value")
        #plt.savefig(" Q2|PART:C: Lang_CDF.png",bbox_inches='tight')
        plt.show()
        # No-Language CDF
        plt.figure()
        plt.bar(x_plot, posterior_cdf_nl, color='r')
        plt.title("Posterior CDF of No-Language")
        plt.xlabel("x_nl * 1000")
        plt.ylabel("Posterior CDF Value")
        #plt.savefig(" Q2|PART:C: NOLang_CDF.png",bbox_inches='tight')
        plt.show()

        # Obtaining the 95% Confidence Bounds for Both Cases
        l_lower, l_upper = conf_interval(posterior_cdf_l)
        nl_lower, nl_upper = conf_interval(posterior_cdf_nl)
        # For x_l
        lang_low, lang_up = x_range[l_lower],x_range[l_upper]
        # For x_nl
        no_lang_low, no_lang_up = x_range[nl_lower], x_range[nl_upper]
        print(" 95% Confidence Bounds for Language Task x_l : Lower = " + str(lang_low) + " & Upper = "+str(lang_up))
        print(" 95% Confidence Bounds for No-Language Task x_nl : Lower = "+str(no_lang_low)+" & Upper = "+str(no_lang_up))

        print("Question 2: Part D")
        # Obtaining joint pdf using the outer product of marginal pdfs
        joint_pdf = np.outer(posterior_Xl, posterior_Xnl)
        # Plotting with imshow
        plt.figure()
        plt.imshow(joint_pdf.T, origin='lower')
        plt.title("Joint Posterior PDF P(X_l, X_nl|A)")
        plt.xlabel("P(X_l|A)*1000")
        plt.ylabel("P(X_nl|A)*1000")
        plt.colorbar()
        #plt.savefig(" Q2|PART:D: Colormap.png",bbox_inches='tight')
        plt.show()

        # P(X_l>X_nl|A) & P(X_l<=X_nl|A)
        # Setting the initial probability totals
        C_bigger = 0
        C_lesser = 0
        for i in range(1001):
            for j in range(1001):
                # Upper Triangle
                if i > j:
                    C_bigger += joint_pdf[i][j]
                # Lower Triangle
                else:
                    C_lesser += joint_pdf[i][j]

        print("P(X_l>X_nl|A) = " + str(C_bigger))
        print("P(X_l<=X_nl|A) = " + str(C_lesser))

        print("Question 2: Part E")
        # Defining P(language) = P(no-language) = 0.5
        P_language = 0.5
        # Estimates from part B
        P_l_max = index_max_l
        P_nl_max = index_max_nl
        # Expressing the hand derived relation for P(activation)
        P_activation = (P_l_max * P_language) + (P_nl_max*P_language)
        # Desired conditional probability then becomes
        lang_given_act = (P_l_max*P_language)/P_activation
        print("Reverse Bayesian Inference leads to: ")
        print("P(language|activation) = " + str(lang_given_act))


# Functions used in the assignment
# Likelihood computation function for Q2:A
def likelihood_a(total, n_active, interval):
    n_passive = total - n_active
    order = np.math.factorial(total)/(np.math.factorial(total-n_active)*np.math.factorial(n_active))
    p_active = interval**n_active
    p_passive = (1-interval)**n_passive
    L = order*p_active*p_passive
    return L

# Finding the 95% confidence interval with the following function
def conf_interval(cdf):
    lower = True
    index = 0
    upper = True
    for i in cdf:
        if i >= 0.025:
            if lower:
                lower_lang = i
                lower_index = index
                lower = False
        if i >= 0.975:
            if upper:
                upper_lang = i
                upper_index = index
                upper = False
        index += 1

    return lower_index,upper_index

arman_budunoglu_21602635_hw1(question)
