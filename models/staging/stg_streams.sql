select
    stream_id,
    seller_size,
    time_budget_s,
    planned_auctions,
    pre_period_gmv
from {{ source('raw', 'raw_streams') }}