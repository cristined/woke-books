# Underrepresented Author Book Recommender

## Mission
What I love about reading is it one of the few ways that we get a chance to see the world from another person's point of view. After reading _The Bluest Eye_ by Toni Morrison, my mind was completely blown and I could not believe I had never read anything like that. So obviously the first thing I did was an analysis on my own reading to break down the number of books I had read by author's race and gender. I saw that a huge percentage of books I read were written by white males and as much as I love everything that Neil Gaiman does it made me wish that I had a tool that could take my GoodReads read book ratings and recommend books that are to my taste but are by authors from more diverse backgrounds. Thus this project was born.

### Data
I used two main data sources:
1. [Amazon product data from Julian McAuley, UCSD](http://jmcauley.ucsd.edu/data/amazon/)
2. [GoodReads API](https://www.goodreads.com/api/index) - using only about the top 10K rated books

GoodReads Author API had gender include for most authors. Ideally the race would have been self reported along with the gender. To categorize the author's race, I enlisted friends to mechanical turk for me using the GoodReads bios and other info along with google and wikipedia. We decided to use the United States EEO race and ethnicity categories. This was not a simple or straight forward task and I am sure we made some mistakes, but in the interest of trying to recommend more diversity in our reading we had to try.

Big thanks to Maivy Nguyen, Rohit Unni, Katie Lazell Fairman, Greg Rafferty, Rebecca Czyrnik, and Moses Marsh for mechanical turking for me.
