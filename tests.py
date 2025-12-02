import pytest
from model import Customer, Product


@pytest.fixture
def customer():
    """Create a basic customer fixture"""
    return Customer(preferences=[5, 5, 5], max_distance=27, price_sensitivty=1.0)


@pytest.fixture
def ref_product():
    """Create a reference product"""
    return Product(price=100, sku="ref_item", category=1, features=[5, 5, 5])


@pytest.fixture
def similar_product():
    """Create a product similar to ref_product but different SKU"""
    return Product(price=110, sku="similar_item", category=1, features=[5, 5, 6])


@pytest.fixture
def different_category_product():
    """Create a product in a different category"""
    return Product(price=120, sku="other_item", category=2, features=[5, 5, 5])


class TestMemoryAccess:
    """Tests for the memory access functionality"""

    def test_no_reference_returns_none(self, customer):
        """When there is no reference at all, it should return None"""
        test_product = Product(price=100, sku="new_item", category=1, features=[5, 5, 5])

        result = customer.access_memory(test_product)

        assert result is None

    def test_direct_reference_match(self, customer, ref_product):
        """When there is a direct reference (same SKU and category), it should return that reference"""
        # Add product to memory
        customer.append_memory(ref_product)

        # Query for the same SKU
        test_product = Product(price=150, sku="ref_item", category=1, features=[5, 5, 5])
        result = customer.access_memory(test_product)

        assert result is not None
        assert len(result) == 1
        assert result[0][0] == 100  # price
        assert result[0][1] == [5, 5, 5]  # features

    def test_memory_reference_similar_product(self, customer, ref_product, similar_product):
        """When there is a reference in the same category (via memory_reference), it should return similar product"""
        # Add reference product to memory
        customer.append_memory(ref_product)

        # Query for similar product (different SKU, same category, similar features)
        test_product = Product(price=105, sku="unknown_item", category=1, features=[5, 5, 4])
        result = customer.access_memory(test_product)

        # Should find the reference product by similarity
        assert result is not None
        assert len(result) == 1
        assert result[0][0] == 100  # ref_product price

    def test_no_reference_different_category(self, customer, ref_product, different_category_product):
        """When product is in different category than memory, it should return None"""
        # Add product in category 1
        customer.append_memory(ref_product)

        # Query for product in category 2
        result = customer.access_memory(different_category_product)

        assert result is None

    @pytest.mark.parametrize(
        "memory_products,query_product,expected_price",
        [
            # Single reference
            (
                [Product(price=100, sku="item1", category=1, features=[5, 5, 5])],
                Product(price=0, sku="item1", category=1, features=[5, 5, 5]),
                100,
            ),
            # Multiple references for same SKU
            (
                [
                    Product(price=100, sku="item1", category=1, features=[5, 5, 5]),
                    Product(price=110, sku="item1", category=1, features=[5, 5, 5]),
                ],
                Product(price=0, sku="item1", category=1, features=[5, 5, 5]),
                100,  # Should return both prices
            ),
        ],
    )
    def test_memory_retrieval_scenarios(
        self, customer, memory_products, query_product, expected_price
    ):
        """Test various memory retrieval scenarios"""
        # Add all memory products
        for product in memory_products:
            customer.append_memory(product)

        # Query memory
        result = customer.access_memory(query_product)

        assert result is not None
        assert result[0][0] == expected_price


class TestMemoryAppend:
    """Tests for appending products to memory"""

    def test_append_creates_category(self, customer):
        """Appending a product should create the category if it doesn't exist"""
        product = Product(price=100, sku="item1", category=1, features=[5, 5, 5])

        customer.append_memory(product)

        assert 1 in customer.memory
        assert "item1" in customer.memory[1]

    def test_append_creates_sku_list(self, customer):
        """Appending a product should create the SKU list if it doesn't exist"""
        product = Product(price=100, sku="item1", category=1, features=[5, 5, 5])

        customer.append_memory(product)

        assert len(customer.memory[1]["item1"]) == 1
        assert customer.memory[1]["item1"][0] == (100, [5, 5, 5])

    def test_append_adds_to_existing_sku(self, customer):
        """Appending multiple products with same SKU should add to the list"""
        product1 = Product(price=100, sku="item1", category=1, features=[5, 5, 5])
        product2 = Product(price=110, sku="item1", category=1, features=[5, 5, 6])

        customer.append_memory(product1)
        customer.append_memory(product2)

        assert len(customer.memory[1]["item1"]) == 2
        assert customer.memory[1]["item1"][0] == (100, [5, 5, 5])
        assert customer.memory[1]["item1"][1] == (110, [5, 5, 6])


class TestMemoryReference:
    """Tests for the memory_reference function (finding similar products)"""

    def test_finds_similar_product_within_threshold(self, customer):
        """Should find similar product when features are within similarity threshold"""
        ref_product = Product(price=100, sku="ref", category=1, features=[5, 5, 5])
        customer.append_memory(ref_product)

        # Query product with similar features
        query_product = Product(price=120, sku="query", category=1, features=[5, 5, 6])
        mem_1 = customer.memory.get(1, {})

        result = customer.memory_refrence(query_product, mem_1, similarity_pct=0.8)

        assert result is not None
        assert result[0][0] == 100  # Should return ref_product

    def test_no_match_outside_threshold(self, customer):
        """Should return None when no product is within similarity threshold"""
        ref_product = Product(price=100, sku="ref", category=1, features=[1, 1, 1])
        customer.append_memory(ref_product)

        # Query product with very different features
        query_product = Product(price=120, sku="query", category=1, features=[9, 9, 9])
        mem_1 = customer.memory.get(1, {})

        result = customer.memory_refrence(query_product, mem_1, similarity_pct=0.8)

        assert result is None

    def test_returns_most_similar_product(self, customer):
        """Should return the most similar product when multiple exist"""
        # Add two products with different similarity levels
        # close_product: features [5,5,6] - distance 1 from query [5,5,5] -> similarity = 1 - 1/27 = 0.963
        # far_product: features [5,7,8] - distance 5 from query [5,5,5] -> similarity = 1 - 5/27 = 0.815
        close_product = Product(price=100, sku="close", category=1, features=[5, 5, 6])
        far_product = Product(price=110, sku="far", category=1, features=[5, 7, 8])

        customer.append_memory(close_product)
        customer.append_memory(far_product)

        # Query for product closer to close_product
        query_product = Product(price=120, sku="query", category=1, features=[5, 5, 5])
        mem_1 = customer.memory.get(1, {})

        result = customer.memory_refrence(query_product, mem_1, similarity_pct=0.8)

        assert result is not None
        # Should return close_product (highest similarity = 0.963)
        assert result[0][0] == 100
