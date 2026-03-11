# payment-service/checkoutService.py
# Handles payment processing, checkout calculations, refund logic

import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLONE GROUP A — "cart total with tax"
# Semantic clone of: order-service/orderService.js::calculateOrderTotal()
# Same logic: iterate items, accumulate subtotal, apply tax rate
# Different language: Python vs JS, different variable names
# ─────────────────────────────────────────────────────────────────────────────
def compute_cart_total(cart_items, tax_rate):
    subtotal = 0
    for item in cart_items:
        subtotal += item['unit_price'] * item['qty']
    tax_amount = subtotal * tax_rate
    return subtotal + tax_amount


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — payment authorization
# No clone. Tests that payment-specific logic stays unclustered.
# ─────────────────────────────────────────────────────────────────────────────
def authorize_payment(card_number, amount, currency):
    logger.info(f'Authorizing payment of {amount} {currency}')
    if amount <= 0:
        raise ValueError('Payment amount must be positive')
    masked = card_number[-4:]
    logger.info(f'Card ending in {masked} authorized')
    return {
        'authorized': True,
        'masked_card': masked,
        'amount': amount,
        'currency': currency,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — refund calculation with tiered fee
# No clone. Domain-specific logic.
# ─────────────────────────────────────────────────────────────────────────────
def calculate_refund_amount(original_amount, days_since_purchase, reason):
    logger.info(f'Calculating refund for amount {original_amount}')
    if days_since_purchase <= 7:
        refund_rate = 1.00
    elif days_since_purchase <= 30:
        refund_rate = 0.85
    else:
        refund_rate = 0.50

    if reason == 'DEFECTIVE':
        refund_rate = 1.00

    refund_amount = original_amount * refund_rate
    fee = original_amount - refund_amount
    logger.info(f'Refund amount: {refund_amount}, processing fee: {fee}')
    return refund_amount


# ─────────────────────────────────────────────────────────────────────────────
# CLONE GROUP D — "paginated list builder"
# Semantic clone of: user-service/userService.js::paginate_results()
# Same logic: slice list by offset/limit, compute metadata
# Different domain: transactions vs users
# ─────────────────────────────────────────────────────────────────────────────
def get_transaction_page(transactions, page, page_size):
    total_count = len(transactions)
    offset = (page - 1) * page_size
    page_items = transactions[offset: offset + page_size]
    total_pages = (total_count + page_size - 1) // page_size
    return {
        'items': page_items,
        'page': page,
        'page_size': page_size,
        'total_count': total_count,
        'total_pages': total_pages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — currency conversion (LONG — tests slicing)
# No clone. Many logging/comment lines around core logic.
# ─────────────────────────────────────────────────────────────────────────────
def convert_currency(amount, from_currency, to_currency, rates):
    logger.info('Starting currency conversion')
    logger.debug(f'Input amount: {amount}')
    logger.debug(f'From: {from_currency} To: {to_currency}')
    logger.debug('Fetching exchange rates from rate table')
    logger.debug('Validating rate table structure')
    logger.debug('Checking rate freshness')
    logger.debug('Rate table validated successfully')
    logger.info('Looking up conversion path')
    logger.debug('Attempting direct conversion')

    if from_currency not in rates:
        logger.error(f'Unknown source currency: {from_currency}')
        raise ValueError(f'Unknown currency: {from_currency}')

    if to_currency not in rates:
        logger.error(f'Unknown target currency: {to_currency}')
        raise ValueError(f'Unknown currency: {to_currency}')

    base_amount = amount / rates[from_currency]
    converted = base_amount * rates[to_currency]
    rounded = round(converted, 2)

    logger.info(f'Conversion complete: {amount} {from_currency} '
                f'= {rounded} {to_currency}')
    logger.debug('Writing conversion audit record')
    logger.debug('Audit record written')
    logger.debug('Conversion context released')

    return rounded