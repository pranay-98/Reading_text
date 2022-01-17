# Reading_text
NLP

- In this question we were asked to implement a solution for recognizing letters in an image. We were asked to solve this problem using simple bayes net and HMM (viterbi algorithm)

- For solving this problem we made a custom class (DetectLetters) which trains on a text dataset and returns the solution for a simple bayes net and viterbi algorithm.

- **Training Phase**

  1. We read the text file and clean that file by removing all the characters which are not valid english language character.
  2. Calculate the initial probabilities. These are the probability of a character being used as the first letter of a sentence in the given text file. We took log of these probabilities for getting rid of numeric underflow problem.
  3. Calculate transition probabilities. These are the probability that a character is followed by another character. We calculate this from the training text file. We took log of these probabilities for getting rid of numeric underflow problem.
  4. Calculate emission probabilities. These are the probability of a test character looking similar to a train character. We calculate this by getting the matching pixels between a train and test pattern and then apply naive bayes on it. We took log of these probabilities for getting rid of numeric underflow problem.

- **Simple Bayes Net**

  - In the simple bayes net for each test character we just return the corresponding train character with the highest emission probability.

- **Viterbi Algorithm**

  - In viterbi algorithm we consider the transition probability, emission probability and probability of best part till that state

  - For the first word in a sentence we calculate initial probability as the probability of that label and multiplied with the emission probability of the word and store that into viterbi table.
  - Then, By iterating over all other words by calculating transition probabilities multiplied with the product of previous viterbi probability stored in that table and emission probability of that word. At each step for each label we store only the max probability obatined by computing the previous equation. Along with that, we also store the label from which this is transitioned.
  - We continue this step till the end of the sentence.
  - From the max probability obtained for the label at the last word we backtrack and get all the labels from where we transitioned and get to the start word and store all the labels into a list.
