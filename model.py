from dataclasses import dataclass, field
import statistics
import math


@dataclass
class Product:
    price: float
    sku: str
    category: int
    features: list[int]


@dataclass
class Customer:

    preferences: list[int]
    max_distance: float
    memory: dict[int, dict[str, list[tuple[float, list[float]]]]] = field(
        default_factory=dict
    )
    price_sensitivty: float = 1.0

    def l1_distance(self, item_1: list[int], item_2: list[int]):

        distance = sum(abs(pf - sp) for pf, sp in zip(item_1, item_2))
        return 1.0 - (distance / self.max_distance)

    def append_memory(self, product: Product):
        # 1. Check if category exists, if not create it
        if product.category not in self.memory:
            self.memory[product.category] = {}

        # 2. Check if SKU exists in that category, if not create list
        if product.sku not in self.memory[product.category]:
            self.memory[product.category][product.sku] = []

        # 3. Append new price and features
        self.memory[product.category][product.sku].append(
            (product.price, product.features)
        )

    def access_memory(self, product: Product):

        mem_1 = self.memory.get(product.category, {})
        if mem_1.get(product.sku):
            return mem_1.get(product.sku)
        elif mem_1:
            return self.memory_refrence(product, mem_1)
        else:
            # We will default to value-based scoring
            return None

    def memory_refrence(
        self,
        product: Product,
        mem_1: dict,
        similarity_pct: float = 0.8,
    ):
        best_similarity = float("-inf")  # Looking for highest similarity
        best_sku = None

        for sku, refs in mem_1.items():
            for _, feature_list in refs:
                # Calculate similarity score (higher = more similar)
                similarity = self.l1_distance(
                    item_1=product.features, item_2=feature_list
                )

                # Track the best match within threshold (higher similarity is better)
                if similarity >= similarity_pct and similarity > best_similarity:
                    best_similarity = similarity
                    best_sku = sku

        # Return the full memory entry for the most similar product
        if best_sku is not None:
            return mem_1[best_sku]

        return None

    def price_refrence(self, prices: list):
        price_ref = statistics.median(prices)
        # Get the absolute deviation
        abs_devs = [abs(x - price_ref) for x in prices]
        # Compute MAD
        mad = statistics.median(abs_devs)
        # Create relative uncertainty
        rel_uncert = mad / price_ref
        return price_ref, rel_uncert

    def eval_product(
        self, product: Product, chart: bool = False, update_mem: bool = True
    ):
        """
        Evaluate a product and return a score.

        WARNING: chart=True is computationally expensive (100+ score calculations per chart).
        Use only for debugging/visualization, not in production.

        In production, always use update_mem=True to maintain customer memory.
        """
        value = self.l1_distance(item_1=product.features, item_2=self.preferences)
        mem_1 = self.access_memory(product)
        score = self.score(mem_1=mem_1, product=product, value=value, chart=chart)

        # Add to memory
        if update_mem:
            self.append_memory(product=product)
        return score

    def score(self, mem_1: dict, product: Product, value: float, chart: bool = False):
        # No reference price available - score purely on value match
        if mem_1 is None or len(mem_1) == 0:
            final_score = value * 100  # 0-100 based on feature match
            return final_score

        # Check price reference from memory
        # Return median and relative uncertainty (mad / median)
        ref, rel_uncert = self.price_refrence([price for price, _ in mem_1])

        # Calibrate baseline so that product with value 0.5
        # is worth reference price
        if value <= 0.5:
            wtp = ref * (value * 2.0)
        else:
            max_premium = 0.6 / self.price_sensitivty
            wtp = ref * (1.0 + max_premium * (value - 0.5))

        # Get the percentage difference
        rel_delta = (wtp - product.price) / wtp

        # If rel_delta between wtp and price is within
        # the relative uncertainty we consider it to
        # be equal to the market price.
        if abs(rel_delta) < rel_uncert:
            rel_delta = 0

        if rel_delta >= 0:
            # If delta is positive (better price)
            # We create a positive feeling with diminishing returns
            feeling = rel_delta**0.65
        else:
            # If delta is negative (worse price)
            # We create a negative feeling with diminishing returns
            # loss aversion: overpriced hurts ~2x
            feeling = -2 * ((-rel_delta) ** 0.65)

        k_eff = 6.0 * self.price_sensitivty

        # Use sigmoid to determine score
        score = 100.0 / (1.0 + math.exp(-k_eff * feeling))
        final_score = max(0.0, min(100.0, score))

        if chart:
            from plotting import plot_customer_position

            plot_customer_position(self, product, final_score, ref, wtp, value)

        return final_score


if __name__ == "__main__":
    from plotting import clear_charts_folder

    # Clear previous charts
    clear_charts_folder()

    # Test the chart generation
    ref_product = Product(price=1, sku="snickers", category=1, features=[5, 5, 5])

    customer = Customer(preferences=[3, 6, 8], max_distance=27, price_sensitivty=1.0)
    customer.append_memory(ref_product)

    test_product_1 = Product(price=0.99, sku="snickers", category=1, features=[5, 5, 5])
    test_product_2 = Product(price=1.20, sku="snickers", category=1, features=[5, 5, 5])

    score_1 = customer.eval_product(test_product_1, chart=True, update_mem=True)
    print(f"Score: {score_1:.2f}")

    score_2 = customer.eval_product(test_product_2, chart=True, update_mem=False)
    print(f"Score: {score_2:.2f}")
