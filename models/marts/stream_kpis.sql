-- Stream grain: the KPI tree rolled up per stream.
-- This is the decision-metric table (GMV per stream) + throughput + pre covariate.
with auc as (
    select * from {{ ref('fct_auctions') }}
)
select
    stream_id,
    variant,
    variant_name,
    any_value(pre_period_gmv)              as pre_period_gmv,
    sum(sale_price)                        as gmv,
    count(*)                               as items_sold,
    avg(sale_price)                        as avg_sale_price,
    avg(duration_s)                        as avg_duration_s,
    avg(n_bids)                            as avg_bids_per_auction,
    sum(case when was_sniped = 1 then 1 else 0 end) as sniped_auctions
from auc
group by stream_id, variant, variant_name