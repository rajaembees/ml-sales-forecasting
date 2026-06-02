import os
import numpy as np
import pandas as pd

POPULARITY_ENCODING = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}

# Coefficients from the notebook's best model (lin_reg3)
_COEF = {
    "Advertising Expenditure": 36.8,
    "Campaign Engagement Score": 20.78,
    "Discount Percentage": -6.92,
    "Average Customer Rating": 2890.28,
    "Product Price": -8.67,
    "Return Rate": -444.79,
    "Length of Product Description": -0.051,
    "Popularity": 146.84,
    "Region_North": -6.98,
    "Region_South": 52.88,
    "Region_West": 189.17,
    "intercept": 469.6,
}


def generate_sales_data(output_path: str, n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sales data matching the notebook's statistical profile."""
    rng = np.random.default_rng(seed)

    def clipped_normal(mean, std, low, high, size):
        out = []
        while len(out) < size:
            batch = rng.normal(mean, std, size * 3)
            out.extend(batch[(batch >= low) & (batch <= high)].tolist())
        return np.array(out[:size])

    regions = rng.choice(["North", "South", "East", "West"], size=n_samples, p=[0.22, 0.26, 0.26, 0.26])
    popularity = rng.choice(
        list(POPULARITY_ENCODING.keys()), size=n_samples, p=[0.02, 0.08, 0.20, 0.30, 0.40]
    )

    adv_exp = clipped_normal(607.7, 82.0, 97.25, 801.5, n_samples)
    campaign_score = clipped_normal(49.5, 13.2, 0.0, 98.75, n_samples)
    discount_pct = clipped_normal(29.0, 4.7, 0.0, 40.54, n_samples)
    avg_rating = clipped_normal(4.41, 0.228, 3.65, 4.87, n_samples)
    product_price = clipped_normal(1434.7, 222.8, 9.78, 2000.56, n_samples)
    return_rate = clipped_normal(1.67, 0.9, 0.03, 4.53, n_samples)
    desc_length = clipped_normal(248.7, 60.8, 42.0, 496.0, n_samples)

    pop_numeric = np.array([POPULARITY_ENCODING[p] for p in popularity])
    region_north = (regions == "North").astype(float)
    region_south = (regions == "South").astype(float)
    region_west = (regions == "West").astype(float)

    sales = (
        _COEF["Advertising Expenditure"] * adv_exp
        + _COEF["Campaign Engagement Score"] * campaign_score
        + _COEF["Discount Percentage"] * discount_pct
        + _COEF["Average Customer Rating"] * avg_rating
        + _COEF["Product Price"] * product_price
        + _COEF["Return Rate"] * return_rate
        + _COEF["Length of Product Description"] * desc_length
        + _COEF["Popularity"] * pop_numeric
        + _COEF["Region_North"] * region_north
        + _COEF["Region_South"] * region_south
        + _COEF["Region_West"] * region_west
        + _COEF["intercept"]
        + rng.normal(0, 1940, n_samples)
    )
    sales = np.clip(sales, 3257.0, 34421.0)

    df = pd.DataFrame(
        {
            "Advertising Expenditure": np.round(adv_exp, 6),
            "Campaign Engagement Score": np.round(campaign_score, 6),
            "Discount Percentage": np.round(discount_pct, 6),
            "Average Customer Rating": np.round(avg_rating, 6),
            "Product Price": np.round(product_price, 6),
            "Return Rate": np.round(return_rate, 6),
            "Length of Product Description": np.round(desc_length, 6),
            "Region": regions,
            "Popularity": popularity,
            "Sales": np.round(sales, 5),
        }
    )

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} rows  →  {output_path}")
    return df
