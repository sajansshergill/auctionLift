select
    auction_id,
    stream_id,
    sale_price,
    duration_s,
    n_bids,
    was_sniped
from {{ source('raw', 'raw_auctions') }}