import  math

def standard_deviation(iterations, r2_scores, average_r2):
    # sqrt(sum((xi - average)^2)/N)
    sumSquared = 0
    for i in range(iterations):
        distance = r2_scores[i] - average_r2 
        squared_distance = distance ** 2
        sumSquared += squared_distance
    
    sumSquared_by_N = sumSquared / iterations

    standard_deviation = math.sqrt(sumSquared_by_N)
    return standard_deviation