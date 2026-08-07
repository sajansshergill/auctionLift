select
    bid_id,
    auction_id,
    stream_id,
    amount,
    seconds_before_end
from {{ source('raw', 'raw_bids') }}