"""
Data Collector Module
=====================
Collects horse racing data from freely available public sources.

Primary sources:
1. Sample/synthetic data generator for immediate testing
2. Web scraper for publicly available racing results

The sample data generator creates realistic horse racing data based on
real-world distributions, making it perfect for developing and testing
the prediction pipeline before connecting to live data sources.
"""

import os
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Realistic sample data generator
# ---------------------------------------------------------------------------

# Pool data for generating realistic races
HORSE_NAMES = [
    "Thunder Bolt", "Silver Arrow", "Golden Spirit", "Midnight Run",
    "Desert Storm", "Ocean Breeze", "Fire Dancer", "Star Gazer",
    "Wild Card", "Iron Will", "Shadow Runner", "Crystal Clear",
    "Storm Chaser", "Lucky Strike", "Noble Spirit", "Dark Knight",
    "Swift Justice", "Royal Flush", "Magic Moment", "Fast Lane",
    "Bright Star", "Wind Rider", "Eagle Eye", "Diamond Rush",
    "Brave Heart", "Silent Night", "Red Baron", "Blue Moon",
    "King's Crown", "Queen's Grace", "Night Fury", "Day Dream",
    "Cloud Nine", "River Dance", "Mountain Peak", "Valley Girl",
    "Sunset Glow", "Dawn Patrol", "Winter Frost", "Summer Heat",
    "Spring Bloom", "Autumn Gold", "Racing Spirit", "Champion's Way",
    "Victory Lane", "Finish Line", "Pole Position", "Leading Edge",
    "Front Runner", "Dark Horse", "Long Shot", "Sure Thing",
    "High Roller", "Big Spender", "Cash Flow", "Pay Day",
    "Jackpot", "Treasure Hunt", "Gold Rush", "Silver Lining",
    "Bronze Medal", "Iron Horse", "Steel Nerve", "Copper Tone",
    "Platinum Star", "Diamond Dust", "Ruby Red", "Emerald Isle",
    "Sapphire Sky", "Pearl Harbor", "Jade Dragon", "Amber Wave",
    "Crimson Tide", "Scarlet Letter", "Ivory Tower", "Ebony Knight",
    "Velvet Touch", "Silk Road", "Cotton Candy", "Leather Bound",
]

JOCKEY_NAMES = [
    "J. Smith", "R. Moore", "L. Dettori", "T. O'Brien", "M. Kinane",
    "P. Dobbs", "S. De Sousa", "W. Buick", "J. Crowley", "D. Tudhope",
    "O. Murphy", "T. Marquand", "R. Havlin", "J. Mitchell", "C. Lee",
    "A. Kirby", "H. Bentley", "D. Egan", "R. Kingscote", "J. Fanning",
    "B. Curtis", "K. Shoemark", "L. Morris", "F. Norton", "G. Baker",
]

TRAINER_NAMES = [
    "A. O'Brien", "J. Gosden", "C. Appleby", "W. Haggas", "R. Varian",
    "M. Johnston", "S. bin Suroor", "A. Balding", "R. Hannon", "H. Palmer",
    "K. Ryan", "T. Dascombe", "R. Beckett", "E. Walker", "D. O'Meara",
    "M. Channon", "C. Hills", "J. Tate", "S. Kirk", "R. Charlton",
]

TRACKS = [
    "Ascot", "Newmarket", "Epsom", "York", "Cheltenham",
    "Goodwood", "Doncaster", "Chester", "Sandown", "Kempton",
    "Lingfield", "Wolverhampton", "Newcastle", "Haydock", "Newbury",
]

GOING_CONDITIONS = [
    "Firm", "Good to Firm", "Good", "Good to Soft", "Soft", "Heavy",
]

RACE_CLASSES = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

RACE_TYPES = ["Flat", "Hurdle", "Chase", "National Hunt Flat"]

DISTANCE_FURLONGS = [5, 6, 7, 8, 10, 12, 14, 16, 20, 24]


def generate_sample_data(
    num_races: int = 1500,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate realistic synthetic horse racing data for model development.

    This creates data with realistic correlations:
    - Better horses (lower base speed) tend to finish higher
    - Going conditions affect different horses differently
    - Jockeys and trainers have skill modifiers
    - Weight carried affects performance
    - Recent form influences results
    """
    logger.info(f"Generating {num_races} synthetic races...")
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    all_entries = []
    race_id = 0
    start_date = datetime(2020, 1, 1)

    # Assign persistent attributes to horses, jockeys, trainers
    horse_ability = {name: np.random.normal(100, 15) for name in HORSE_NAMES}
    jockey_skill = {name: np.random.normal(0, 3) for name in JOCKEY_NAMES}
    trainer_skill = {name: np.random.normal(0, 2) for name in TRAINER_NAMES}

    # Horse-distance and horse-going preferences
    horse_dist_pref = {
        name: random.choice(DISTANCE_FURLONGS) for name in HORSE_NAMES
    }
    horse_going_pref = {
        name: random.choice(GOING_CONDITIONS) for name in HORSE_NAMES
    }

    for i in range(num_races):
        race_id += 1
        race_date = start_date + timedelta(days=random.randint(0, 1800))
        track = random.choice(TRACKS)
        going = random.choice(GOING_CONDITIONS)
        race_class = random.choice(RACE_CLASSES)
        race_type = random.choice(RACE_TYPES)
        distance = random.choice(DISTANCE_FURLONGS)
        num_runners = random.randint(5, 16)
        prize_money = random.choice([5000, 10000, 15000, 25000, 50000, 100000])

        # Select runners for this race
        runners = random.sample(HORSE_NAMES, min(num_runners, len(HORSE_NAMES)))

        # Calculate performance scores for each runner
        performances = []
        for horse in runners:
            jockey = random.choice(JOCKEY_NAMES)
            trainer = random.choice(TRAINER_NAMES)
            age = random.randint(2, 9)
            weight_lbs = random.randint(112, 168)
            draw = random.randint(1, num_runners)
            odds = round(random.uniform(1.5, 50.0), 1)

            # Performance model
            base = horse_ability[horse]
            perf = base + jockey_skill[jockey] + trainer_skill[trainer]

            # Distance preference effect
            dist_diff = abs(distance - horse_dist_pref[horse])
            perf -= dist_diff * 0.8

            # Going preference effect
            going_idx = GOING_CONDITIONS.index(going)
            pref_idx = GOING_CONDITIONS.index(horse_going_pref[horse])
            going_diff = abs(going_idx - pref_idx)
            perf -= going_diff * 2.0

            # Weight penalty
            perf -= (weight_lbs - 126) * 0.15

            # Age effect (prime years 4-6)
            age_penalty = abs(age - 5) * 1.5
            perf -= age_penalty

            # Random race-day variation
            perf += np.random.normal(0, 8)

            performances.append(
                {
                    "race_id": f"R{race_id:05d}",
                    "race_date": race_date.strftime("%Y-%m-%d"),
                    "track": track,
                    "going": going,
                    "race_class": race_class,
                    "race_type": race_type,
                    "distance_furlongs": distance,
                    "prize_money": prize_money,
                    "num_runners": num_runners,
                    "horse_name": horse,
                    "jockey": jockey,
                    "trainer": trainer,
                    "age": age,
                    "weight_lbs": weight_lbs,
                    "draw": draw,
                    "odds": odds,
                    "performance_score": perf,
                    # Pre-race ability estimate (used to derive realistic
                    # odds *before* the race outcome is known).  Uses the
                    # deterministic part of the performance model — the
                    # race-day noise is intentionally excluded so odds
                    # carry genuine uncertainty.
                    "_pre_race_rating": (
                        base
                        + jockey_skill[jockey]
                        + trainer_skill[trainer]
                        - dist_diff * 0.8
                        - going_diff * 2.0
                        - (weight_lbs - 126) * 0.15
                        - age_penalty
                    ),
                }
            )

        # Sort by performance (highest = best) and assign finish positions
        performances.sort(key=lambda x: x["performance_score"], reverse=True)

        # --- Generate realistic pre-race odds from ability ratings ----
        # Odds are derived from the *pre-race* rating (no race-day noise),
        # converted to a probability via softmax, then to decimal odds
        # with a bookmaker overround and some noise.
        ratings = np.array([p["_pre_race_rating"] for p in performances])
        # Softmax to implied probs (higher rating ≈ shorter odds)
        exp_r = np.exp((ratings - ratings.max()) / 10.0)  # temperature=10
        raw_probs = exp_r / exp_r.sum()
        # Apply overround (~120 %) and per-runner noise
        overround = 1.20 + np.random.uniform(-0.05, 0.10)
        noisy_probs = raw_probs * overround + np.random.exponential(0.02, size=len(raw_probs))
        noisy_probs = np.clip(noisy_probs, 0.005, 0.95)
        raw_odds = 1.0 / noisy_probs
        raw_odds = np.clip(raw_odds, 1.2, 100.0)

        for idx, entry in enumerate(performances):
            entry["odds"] = round(float(raw_odds[idx]), 1)

        for pos, entry in enumerate(performances, 1):
            entry["finish_position"] = pos
            entry["won"] = 1 if pos == 1 else 0

            # Generate realistic finishing times
            base_time = distance * 12.5  # ~12.5 seconds per furlong
            entry["finish_time_secs"] = round(
                base_time + (pos - 1) * random.uniform(0.3, 2.0), 2
            )

            # Lengths behind winner
            entry["lengths_behind"] = round((pos - 1) * random.uniform(0.5, 3.0), 1)

            # Remove the internal performance score from the dataset
            del entry["performance_score"]
            del entry["_pre_race_rating"]

        all_entries.extend(performances)

    df = pd.DataFrame(all_entries)

    # Sort by date, then race_id, then finish position
    df = df.sort_values(
        ["race_date", "race_id", "finish_position"]
    ).reset_index(drop=True)

    if save:
        output_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} entries from {num_races} races to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Web scraper for free public racing data
# ---------------------------------------------------------------------------

class RacingDataScraper:
    """
    Scrapes publicly available horse racing results from free websites.
    Uses respectful scraping practices (delays, user-agent identification).
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})

    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a page with rate limiting and error handling."""
        try:
            time.sleep(config.REQUEST_DELAY)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def scrape_racing_post_results(
        self,
        date: str,
    ) -> list[dict]:
        """
        Scrape race results from a public racing results page.

        NOTE: Web scraping is fragile and site-specific. This is a template
        that demonstrates the approach. You may need to adjust selectors
        based on the target website's current structure.

        Args:
            date: Date string in YYYY-MM-DD format
        """
        results = []
        url = f"https://www.racingpost.com/results/{date}"
        logger.info(f"Scraping results for {date} from {url}")

        soup = self._get_page(url)
        if not soup:
            return results

        # NOTE: Selectors below are illustrative; actual selectors depend
        # on the website's current HTML structure and may need updating.
        race_cards = soup.find_all("div", class_="rp-raceTimeCourseName")

        for card in race_cards:
            try:
                track_elem = card.find("a", class_="rp-raceTimeCourseName__name")
                track = track_elem.text.strip() if track_elem else "Unknown"

                rows = card.find_next("table").find_all("tr")
                for row in rows[1:]:  # Skip header
                    cols = row.find_all("td")
                    if len(cols) >= 5:
                        results.append(
                            {
                                "race_date": date,
                                "track": track,
                                "finish_position": cols[0].text.strip(),
                                "horse_name": cols[1].text.strip(),
                                "jockey": cols[2].text.strip(),
                                "trainer": cols[3].text.strip(),
                                "odds": cols[4].text.strip(),
                            }
                        )
            except Exception as e:
                logger.warning(f"Error parsing race card: {e}")
                continue

        return results

    def scrape_date_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Scrape results for a range of dates.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        all_results = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            day_results = self.scrape_racing_post_results(date_str)
            all_results.extend(day_results)
            logger.info(
                f"  {date_str}: {len(day_results)} results collected"
            )
            current += timedelta(days=1)

        df = pd.DataFrame(all_results)
        if not df.empty:
            output_path = os.path.join(
                config.RAW_DATA_DIR, "scraped_results.csv"
            )
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} scraped results to {output_path}")

        return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_data(
    source: str = "sample",
    num_races: int = 1500,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Main function to collect horse racing data.

    Args:
        source: "sample" for synthetic data, "scrape" for web scraping
        num_races: Number of races for synthetic data generation
        start_date: Start date for scraping (YYYY-MM-DD)
        end_date: End date for scraping (YYYY-MM-DD)

    Returns:
        DataFrame with race results
    """
    if source == "sample":
        return generate_sample_data(num_races=num_races)
    elif source == "scrape":
        scraper = RacingDataScraper()
        return scraper.scrape_date_range(start_date, end_date)
    else:
        raise ValueError(f"Unknown data source: {source}")


if __name__ == "__main__":
    # Generate sample data for testing
    df = collect_data(source="sample", num_races=1500)
    print(f"\nGenerated dataset shape: {df.shape}")
    print(f"\nSample data:\n{df.head(10)}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
