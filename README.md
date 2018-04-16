# Woke Books
a.k.a Underrepresented Author Book Recommender

## Mission
Reading is one of the best tools we have to try to see the world from a perspective that is different from our own. Recommendation engines are often echo chambers suggesting items solely on their similarity to what you have previously read. I wanted to create a book recommender that would show me books that are not just like the books I have read before, but are in the genres I am interested in but by authors with different perspectives than I normally read and help expand my worldview.

## Process

### Data
I used three data sources:
1. [Amazon product data from Julian McAuley, UCSD](http://jmcauley.ucsd.edu/data/amazon/) - for text reviews
2. [Goodreads API](https://www.goodreads.com/api/index) - for book and author data
3. [Goodbooks 10k](https://github.com/zygmuntz/goodbooks-10k) - for ratings and kicking off point of top 10K ranked books on GoodReads

GoodReads Author API had gender reported for most authors. Ideally the race would have been self reported along with the gender. To categorize the author's race, I enlisted friends to mechanical turk for me using the GoodReads bios and other info along with google and wikipedia. We decided to use the United States EEO race and ethnicity categories. This was not a simple or straight forward task and I am sure we made some mistakes, but in the interest of trying to recommend more diversity in our reading we had to try.

Big thanks to Maivy Nguyen, Rohit Unni, Katie Lazell Fairman, Greg Rafferty, Rebecca Czyrnik, and Moses Marsh for mechanical turking for me.

### Collaborative Filtering on Apache Spark
Using 6.5 million reviews by over 50 thousand users on goodreads, I decided to only use the goodreads reviews for this piece and not include the amazon reviews because the average books per user in goodreads was about 111 and in Amazon it was about 4 and the ratings distribution was very different for the two sources. People tended to give higher reviews on amazon when they are in the frame of mind to buy or not buy compared to gooodreads where they are comparing it to all the other books they have read and get a little more stingy with the 5's. I used ALS in Spark to create a user factors matrix and an book factor matrix. For existing users you are able to multiply the user factor and the book factor row to get a recommendation for how that user would like a certain book.

When a new user joins goodreads they rate books they have read in the past and I would like to be able to give them recommendations without refactoring the entire collaborative filter recommender. I used the book factor matrix and the user's ratings to do gradient descent to create a user factor vector which can be used with the book factor matrix to make predictions on how the user would rate unread books, which the program could then recommend.

### Gradient Descent into Madness
The most difficult part of trying to find the user vector by gradient descent was finding the best metric to measure how successful the recreation was in comparison to how the user vector would look if the program refactored the entire matrix in Spark. The metric that ended up making the most sense was comparing the book rating rank similarity of what the gradient descent vector’s dot product with the item matrix was compared to the Spark user matrix dotted with the item matrix.

I needed to gridsearch not only what is the best performing recommender while testing in Spark but what rank for the item matrix creates the best recreation during gradient descent. I split the ratings data by users into 3 different groups: training, validation, and testing.

#### Spark Training
In Spark I split the training user group into a train and test group and ran a grid search for different regularization parameters for different ranks. Once I had optimized for my test ranks, I ran and saved the book and user matrices for both the training users and the training/validation users.

In Spark, a low rank matrix was able to produce very good results but once the matrices were passed to do gradient descent for new users, a lot of the user's taste were lost in translation. At the end of this step there were 4 matrices for each rank we were testing: training only user matrix, training only books matrix, training/validation user matrix, training/validation book matrix.

#### New User Gradient Descent Training
I took the user ratings from the validation data and used that to do a gradient descent with the training only books matrix to create a user vector. I compared the ranking result of the user vector from gradient descent multiplied with the training only books matrix to the ranking result of that user's vector from training/validation user matrix multiplied with training/validation book matrix. I did a rank similarity comparison because the order was more important than the actual predicted rating.

I tested the same subset of the validation users across different gradient descent parameters and across the different ranks of matrices output by Spark and found that we needed a rank that was approximately 4 times higher to retain individuality with gradient descent.

### Reviews Clustering
Both in the Spark only model and the gradient descent new user model, certain books tend get an unintentional boost that changes based on what rank they are in. I wanted to have the categories of the books so that I could show the user books in categories they were most interested in.

For this step, I aggregated the Amazon reviews per book. I wasn't interested in the individual's feelings about the book but about how this book as a whole compared to other books. This required translating Amazon's different ASIN's which for books should be their 10 character ISBN to the gooodreads best book id.

Once I had aggregated reviews from all the books I was using in the book matrices. I used k-means clustering to create the categories of the review, the 13 clusters seemed to create distinct genres without having any single author categories.

### Get User and Make Recommendations

Using the GoodReads user ID, we are able to get all the users rated books and combine those with the books in the top 10K so we can see which race/gender groups are underrepresented in the users reading and create a user factor vector by using gradient descent to produce recommendations. Since people of color have written less than 10% of books on goodreads I decided not just to use the current percentages as targets. To create targets I changed the numbers so that each race/gender group had written one thousand more books and then I compared the percentage the user has actually read to these target percentages and created a score between 0 and 1 to be added to the users predicted rating of the book. If a user had read almost nothing by that race/gender group books by those authors would get a boost of very close to one star and if they had read almost everything by one race or gender group then the predicted ratings would get almost no boost on the predicted rating.

To further personalize the recommendations of the users, I took the 5 categories that the user had read the largest percentage of that category (since they are not uniform categories) and displayed the top recommendations in each category and then the top recommendations in none of those categories.

It is hard to measure the success of recommendations. To do so within the context of this project I used the goodreads accounts and brains of Catherine Goold, Moses Marsh, Rohit Unni, and Tomas Bielskis to confirm they were getting reasonable recommendations. Ideally I would like to be able to measure the click through rate on these recommendations so I could find if this would spike their initial interest, whether they marked these books as "to-read", and if they read the book within a certain time period.

### Future Work
I would like to continue to improve on the recommender by including user data, time of ratings, and looking deeper into the text/descriptions of the book instead of using reviews alone to create the genres.

Manual classification of the authors was a big limiting factor for scope. Ideally the recommender could expand beyond 10k books, hopefully this would open up the pool of books that would get recommended and boosted up for the user. Also I would like to include LGBTQ and country of origin in the diversity boost score.

## Example

Okay so now you get the theory, but let's see an example

### Moses (Standard Nerd)

Let me introduce you to our sample user Mr. Moses Marsh our dear instructor who is basic af in his nerdiness.

![Standard Nerd](https://github.com/cristined/woke-books/blob/master/img/standard_nerd.png)

So here are his recommendations prior to boosting, I really believe that he could find all of these books himself, in fact he has read some of these and had yet to rate them in goodreads.

![Standard Nerd Recs - Before Boosting](https://github.com/cristined/woke-books/blob/master/img/standard_nerd_recs_before_boosting.png)

Let's look at the current breakdown of authors read.

![Standard Nerd Recs - Books Read](https://github.com/cristined/woke-books/blob/master/img/standard_nerd_books_read.png)

Yikes that is a lot of white males. Using these percentages, the recommender creates a boost between 0 and 1, so for Neil Gaiman he will get a low boost (close to 0) and for Toni Morrison she will get a really high boost (close to 1). So now books by underrepresented demographics with already high ratings will display for the user.

![Standard Nerd Recs - After Boosting](https://github.com/cristined/woke-books/blob/master/img/standard_nerd_recs_after_boosting.png)

Now doesn't this look a little more interesting. Happy reading Moses!
