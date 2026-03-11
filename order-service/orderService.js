// order-service/orderService.js
// Handles order lifecycle: creation, pricing, validation, status management

const logger = require('./logger');

// ─────────────────────────────────────────────────────────────────────────────
// CLONE GROUP A — "cart total with tax"
// Semantic clone of: payment-service/checkoutService.py::compute_cart_total()
// Same logic: iterate items, accumulate subtotal, apply tax rate
// Different language: JS vs Python, different variable names
// ─────────────────────────────────────────────────────────────────────────────
function calculateOrderTotal(orderItems, taxRate) {
    let subtotal = 0;
    for (const item of orderItems) {
        subtotal += item.price * item.quantity;
    }
    const tax = subtotal * taxRate;
    return subtotal + tax;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLONE GROUP B — "percentage discount"
// Semantic clone of: user-service/loyaltyService.java::applyMemberDiscount()
// Same logic: compute discount amount, subtract from original
// Different language: JS vs Java, different variable names
// ─────────────────────────────────────────────────────────────────────────────
function applyDiscountCode(originalPrice, discountPercent) {
    const discountAmount = originalPrice * (discountPercent / 100);
    const finalPrice = originalPrice - discountAmount;
    return finalPrice;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLONE GROUP C — "input validation pattern"
// Semantic clone of: inventory-service/stockService.py::validate_restock_request()
// Same logic: check required fields, accumulate errors, return validity flag
// Different domain names but identical validation pattern
// ─────────────────────────────────────────────────────────────────────────────
function validateOrderRequest(order) {
    const errors = [];

    if (!order.customerId || order.customerId === '') {
        errors.push('customerId is required');
    }
    if (!order.items || order.items.length === 0) {
        errors.push('order must contain at least one item');
    }
    if (order.shippingAddress === null || order.shippingAddress === undefined) {
        errors.push('shippingAddress is required');
    }
    if (order.paymentMethod === null || order.paymentMethod === undefined) {
        errors.push('paymentMethod is required');
    }

    const isValid = errors.length === 0;
    return { isValid, errors };
}

// ─────────────────────────────────────────────────────────────────────────────
// UNIQUE — order status state machine
// No clone in other services. Tests non-clone scoring stays low.
// ─────────────────────────────────────────────────────────────────────────────
function transitionOrderStatus(currentStatus, event) {
    const transitions = {
        'PENDING':    { 'PAYMENT_RECEIVED': 'CONFIRMED', 'CANCEL': 'CANCELLED' },
        'CONFIRMED':  { 'SHIPPED': 'SHIPPED',            'CANCEL': 'CANCELLED' },
        'SHIPPED':    { 'DELIVERED': 'DELIVERED' },
        'DELIVERED':  {},
        'CANCELLED':  {},
    };

    const allowed = transitions[currentStatus];
    if (!allowed) {
        throw new Error(`Unknown status: ${currentStatus}`);
    }
    const nextStatus = allowed[event];
    if (!nextStatus) {
        throw new Error(
            `Invalid transition: ${currentStatus} + ${event}`);
    }
    return nextStatus;
}

// ─────────────────────────────────────────────────────────────────────────────
// UNIQUE — order summary builder (LONG — tests slicing)
// No semantic clone. Many logging lines → should trigger slicing.
// ─────────────────────────────────────────────────────────────────────────────
function buildOrderSummary(order, customer, shippingDetails) {
    logger.info('Building order summary');
    logger.debug('Order ID: ' + order.id);
    logger.debug('Customer ID: ' + customer.id);
    logger.debug('Customer name: ' + customer.name);
    logger.debug('Customer email: ' + customer.email);
    logger.debug('Shipping carrier: ' + shippingDetails.carrier);
    logger.debug('Shipping address: ' + shippingDetails.address);
    logger.debug('Shipping ETA: ' + shippingDetails.eta);
    logger.info('Computing item list...');
    logger.debug('Iterating order items for summary');

    let itemCount = 0;
    let totalValue = 0;

    for (const item of order.items) {
        logger.debug('Adding item: ' + item.name);
        itemCount += 1;
        totalValue += item.price * item.quantity;
    }

    logger.info('Item count computed: ' + itemCount);
    logger.debug('Formatting currency values');
    logger.debug('Attaching shipping details to summary');
    logger.debug('Attaching customer details to summary');
    logger.info('Order summary build complete');
    logger.debug('Returning summary object');

    return {
        orderId:       order.id,
        customerName:  customer.name,
        customerEmail: customer.email,
        itemCount,
        totalValue,
        carrier:       shippingDetails.carrier,
        eta:           shippingDetails.eta,
    };
}

module.exports = {
    calculateOrderTotal,
    applyDiscountCode,
    validateOrderRequest,
    transitionOrderStatus,
    buildOrderSummary,
};