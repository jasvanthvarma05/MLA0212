# Function to calculate posterior probability using Bayes' Theorem
def bayes_theorem(prior, likelihood, false_positive_rate):
    # Calculate the marginal likelihood
    marginal_likelihood = (likelihood * prior) + (false_positive_rate * (1 - prior))
    
    # Calculate the posterior probability
    posterior = (likelihood * prior) / marginal_likelihood
    
    return posterior

# Define the probabilities
prior = 0.01                  # P(H)
likelihood = 0.99             # P(E|H)
false_positive_rate = 0.05    # P(E|¬H)

# Calculate the posterior probability
result = bayes_theorem(prior, likelihood, false_positive_rate)

# Print the result
print(f"The posterior probability of having the disease given a positive test result is: {result:.4f}")
