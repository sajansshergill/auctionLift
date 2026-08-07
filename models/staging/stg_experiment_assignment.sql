select
    stream_id,
    variant,                         -- 1 = soft close (treatment), 0 = hard close
    case when variant = 1 then 'soft_close' else 'hard_close' end as variant_name
from {{ source('raw', 'raw_experiment_assignment') }}