# Experiment Memo: Soft-Close Auction Timer

## Recommendation
Do not ship soft close as a global default.

Soft close increases winning price per auction, but it also makes auctions run longer. In a fixed-length live stream, that throughput loss is larger than the price gain, so GMV per stream falls.

## What We Tested
We simulated a stream-randomized experiment where treatment streams used a soft-close rule: when a bid landed in the final seconds, the auction timer extended. Control streams kept the current hard-close rule.

The decision metric is GMV per stream because streams are the seller session and the unit of randomization. Price per auction is a mechanism metric, not the launch metric.

## Result
For the seeded 2,000-stream run:

- GMV per stream: -3.68% with a 95% CI of [-5.30%, -2.07%].
- Winning price per auction: +1.66% with a 95% CI of [+0.90%, +2.41%].
- Items sold per stream: -6.34%.
- CUPED reduced decision-metric variance by 88.3%.

The experiment validity gate passed: no sample-ratio mismatch, balanced pre-period GMV, no orphaned records, and no invalid prices or bid amounts.

## Interpretation
The buyer behavior mechanism works: late-bid extensions create more price competition. The marketplace outcome does not: sellers sell fewer items in the same stream time budget, and that lost throughput more than offsets the higher prices.

This is a good example of why the north-star metric must match the product surface. Optimizing only for auction price would recommend the wrong launch decision.

## Next Step
Run a targeted follow-up on streams where time is not the binding constraint, such as high-demand inventory, longer planned streams, or sellers with fewer planned auctions. Soft close may still be valuable in segments where the price lift is not paid for with lost throughput.
