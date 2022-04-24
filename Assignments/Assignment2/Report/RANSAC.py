def RANSAC_fit(X,n):
    t = 1
    N = np.inf
    Bestfit = None
    Inliers = []
    sample_points =[] 
    p = 0.99 #probability 𝑝, at least one random sample is free from outliers 
    iterations = 0
    min_s = 3 #Minimum no of points required to find the equation of the circle
    e = 0.5 #Outlier ratio 0.5 was taken for the worst case
    N = np.log(1-p)/np.log(1-(1-e)**min_s)  #calculation of samples
    while N > iterations:
        random_indices= np.random.randint(n, size=min_s)
        point1, point2, point3 = X[random_indices]
        
        #Calculation of the center coordinates and the radius of the circle passing through the sample points
        coefficientMatrix = np.array([[point2[0] - point1[0], point2[1] - point1[1]], [point3[0] - point1[0], point3[1] - point1[1]]]) 
        constantMatrix = np.array([[point2[0]**2 - point1[0]**2 + point2[1]**2 - point1[1]**2], [point3[0]**2 - point1[0]**2 + point3[1]**2 - point1[1]**2]])		
        invCoefficientMatrix = np.linalg.pinv(coefficientMatrix)

        center_x, center_y = (invCoefficientMatrix@constantMatrix) / 2
        center_x, center_y = center_x[0], center_y[0]
        r = np.sqrt((point1[0]- center_x)**2 + (point1[1] - center_y)**2)

        Inlier_test = []
        #Checking for inliers and appending them into the inlier set.
        for x, y in X:
            dis = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if (np.abs(dis - r) < t):
                Inlier_test.append([x,y])

        #Checking whether the number of current inliers is greater than the past inliers
        if (len(Inlier_test) > len(Inliers)):
            Bestfit = [center_x, center_y, r] #Getting the center coordinates and radius of the best fit circle
            Inliers = Inlier_test
            sample_points = [point1, point2, point3]  #Collecting the sample points of the best fit
        iterations+=1
    return Bestfit, Inliers, sample_points
