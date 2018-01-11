# Underrepresented Author Book Recommender

## Mission


### Data
I used two main data sources:
1. [Amazon product data from Julian McAuley, UCSD](http://jmcauley.ucsd.edu/data/amazon/) - for ratings and text reviews
2. [GoodReads API](https://www.goodreads.com/api/index) - for book and author data (using only about the top 9K rated books in english)

GoodReads Author API had gender reported for most authors. Ideally the race would have been self reported along with the gender. To categorize the author's race, I enlisted friends to mechanical turk for me using the GoodReads bios and other info along with google and wikipedia. We decided to use the United States EEO race and ethnicity categories. This was not a simple or straight forward task and I am sure we made some mistakes, but in the interest of trying to recommend more diversity in our reading we had to try.

Big thanks to Maivy Nguyen, Rohit Unni, Katie Lazell Fairman, Greg Rafferty, Rebecca Czyrnik, and Moses Marsh for mechanical turking for me.
