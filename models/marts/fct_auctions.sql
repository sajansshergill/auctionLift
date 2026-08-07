-- Auction grain, enriched with the stream's variant and pre-period covariate.
-- This is what the auction-level price analysis (clustered SE) reads.
select
    a.auction_id,
    a.stream_id,
    x.variant,
    x.variant_name,
    a.sale_price,
    a.duration_s,
    a.n_bids,
    a.was_sniped,
    s.pre_period_gmv
from {{ ref('stg_auctions') }} a
join {{ ref('stg_experiment_assignment') }} x using (stream_id)
join {{ ref('stg_streams') }} s using (stream_id)