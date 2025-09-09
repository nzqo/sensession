"""
The campaign module is meant for longer running experiments. It provides helpers to
run multiple experiments, detect failures and store them for reruns, etc.
"""

from sensession.campaign.runner import (
    CampaignRunner,
    extract_failed_campaign,
    extract_succeeded_campaign,
)
from sensession.campaign.campaign import (
    Campaign,
    SerializationMode,
    write_to_disk,
    read_from_disk,
)
from sensession.campaign.schedule import Schedule, ScheduleBuilder
from sensession.campaign.processor import CampaignProcessor
