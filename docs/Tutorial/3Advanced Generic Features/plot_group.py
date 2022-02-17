"""
Best group subset selection
================================================
"""
# Best group subset selection (BGSS) aims to choose a small part of non-overlapping groups to achieve the best interpretability on the response variable. BGSS is practically useful for the analysis of ubiquitously existing variables with certain group structures. For instance, a categorical variable with several levels is often represented by a group of dummy variables. Besides, in a nonparametric additive model, a continuous component can be represented by a set of basis functions (e.g., a linear combination of spline basis functions). Finally, specific prior knowledge can impose group structures on variables. A typical example is that the genes belonging to the same biological pathway can be considered as a group in the genomic data analysis.
# 
# The BGSS can be achieved by solving:
# 
# ..math::
#     \min_{\beta\in \mathbb{R}^p} \frac{1}{2n} ||y-X\beta||_2^2,\quad s.t.\ ||\beta||_{0,2}\leq s .
# 
# 
# where $||\beta||_{0,2} = \sum_{j=1}^J I(||\beta_{G_j}||_2\neq 0)` in which $||\cdot||_2` is the $L_2` norm and model size $s` is a positive integer to be determined from data. Regardless of the NP-hard of this problem, Zhang et al develop a certifiably polynomial algorithm to solve it. This algorithm is integrated in the `abess` package, and user can handily select best group subset by assigning a proper value to the `group` arguments:
# 
# We still use the dataset `dt` generated before, which has 100 samples, 5 useful variables and 15 irrelevant varibales.



print('real coefficients:\n', dt.coef_, '\n')


# Support we have some prior information that every 5 variables as a group:



group = np.linspace(0, 3, 4).repeat(5)
print('group index:\n', group)


# Then we can set the `group` argument in function. Besides, the `support_size` here indicates the number of groups, instead of the number of variables.



model = LinearRegression(support_size = range(0, 3))
model.fit(dt.x, dt.y, group = group)
print('coefficients:\n', model.coef_)

#%%
# The fitted result suggest that only two groups are selected (since `support_size` is from 0 to 2) and the selected variables are shown before.
# 
# R tutorial
# -----------------------
# For R tutorial, please view [https://abess-team.github.io/abess/articles/v07-advancedFeatures.html](https://abess-team.github.io/abess/articles/v07-advancedFeatures.html).
