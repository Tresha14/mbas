# Checking our scores against human ratings

`tools/validate_ratings.py` answers "does the score match what people think?". Ported from the older project.

1. Record several interviews and open each report once so it is built (about 10 or more for a result you can quote).
2. At least two people, who have not seen our scores, rate each interview from 1 to 5 on overall, content, fluency,
   composure, delivery, eye contact and responsiveness.
3. Write the ratings in a CSV (`python -m tools.validate_ratings --template ratings.csv` makes an empty one).
4. Run `python -m tools.validate_ratings ratings.csv` (add `--json results.json` to keep the numbers).

It prints, per dimension: how well the raters agree with each other (Cohen's kappa, 0.6 or more is treated as
reliable) and how closely our score follows the mean human rating (Pearson and Spearman correlation).

Rules it follows: a statistic that cannot be computed (one interview, a constant series, no disagreement to measure)
prints "undefined", never 0 or 1; fewer than 10 rated interviews prints a warning that the numbers are preliminary;
and if the raters do not agree with each other it says so, because then the human mean is not a trustworthy yardstick.
No real ratings have been collected yet, so there are no results to quote.
