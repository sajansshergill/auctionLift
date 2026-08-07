select
    auction_id,
    stream_id,
    amount
from {{ source('raw', 'raw_purchases') }}