import polars as pl
from polars.testing import assert_frame_equal

from sensession.campaign import CampaignProcessor


def test_sequence_nums_align():
    # fmt: off
    df = pl.LazyFrame({# Session : | 1      |    2    |       3      |       4      |       5       |
        "receiver_name"          : ["A", "A", "B", "A", "A", "B", "B", "B", "C", "B", "A", "B", "C" ],
        "sequence_number"        : [ 1,   3,   2,   1,   1,   1,   1,   1,   3,   3,   1,   1,   2  ],
        "collection_name"        : ["1", "1", "2", "2", "3", "3", "3", "4", "4", "4", "5", "5", "5" ],
        "stream_capture_num"     : [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12 ],
		}
	)

    bf = pl.LazyFrame({# Session : | 1      |    4     |
        "receiver_name"          : ["A", "A", "C", "B" ],
        "sequence_number"        : [ 1,   3,   3,   3, ],
        "collection_name"        : ["1", "1", "4", "4" ],
        "stream_capture_num"     : [ 0,   1,    8,   9 ],
		}
	)
    # fmt: on

    # NOTE: In the above, only the following should remain after aligning:
    # All of collection 1, since A is the only receiver
    # Sequence number 3 of session four, since that is where both participants agree

    proc = CampaignProcessor()
    a = proc._align_sequence_nums(df)
    assert_frame_equal(bf, a)
