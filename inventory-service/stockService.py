# inventory-service/stockService.py
# Handles stock levels, restock logic, warehouse operations

import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLONE GROUP C — "input validation pattern"
# Semantic clone of: order-service/orderService.js::validateOrderRequest()
# Same logic: check required fields, accumulate errors, return validity flag
# Different domain: restock request vs order request
# ─────────────────────────────────────────────────────────────────────────────
def validate_restock_request(request):
    errors = []

    if not request.get('product_id') or request['product_id'] == '':
        errors.append('product_id is required')
    if not request.get('quantity') or request['quantity'] <= 0:
        errors.append('quantity must be a positive number')
    if request.get('warehouse_id') is None:
        errors.append('warehouse_id is required')
    if request.get('supplier_id') is None:
        errors.append('supplier_id is required')

    is_valid = len(errors) == 0
    return {'is_valid': is_valid, 'errors': errors}


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — stock reservation with rollback
# No clone. Complex transactional logic unique to inventory.
# ─────────────────────────────────────────────────────────────────────────────
def reserve_stock(product_id, requested_qty, current_stock, reserved_stock):
    logger.info(f'Reserving {requested_qty} units of {product_id}')
    available = current_stock - reserved_stock
    if available < requested_qty:
        logger.warning(f'Insufficient stock: available={available}')
        return {'success': False, 'available': available}
    new_reserved = reserved_stock + requested_qty
    logger.info(f'Reserved successfully. New reserved: {new_reserved}')
    return {'success': True, 'new_reserved': new_reserved}


# ─────────────────────────────────────────────────────────────────────────────
# CLONE GROUP E — "weighted average calculation"
# Semantic clone of: notification-service/analyticsService.java::computeWeightedScore()
# Same logic: multiply values by weights, sum, divide by total weight
# Different domain: stock price vs notification score
# ─────────────────────────────────────────────────────────────────────────────
def calculate_weighted_average_cost(purchases):
    total_cost = 0
    total_quantity = 0
    for purchase in purchases:
        total_cost += purchase['unit_cost'] * purchase['quantity']
        total_quantity += purchase['quantity']
    if total_quantity == 0:
        return 0
    weighted_avg = total_cost / total_quantity
    return weighted_avg


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — stock level categorization
# No clone. Business-rule specific to inventory thresholds.
# ─────────────────────────────────────────────────────────────────────────────
def categorize_stock_level(current_stock, reorder_point, max_capacity):
    if current_stock == 0:
        return 'OUT_OF_STOCK'
    if current_stock <= reorder_point:
        return 'LOW'
    if current_stock <= reorder_point * 2:
        return 'MEDIUM'
    if current_stock >= max_capacity * 0.9:
        return 'NEAR_FULL'
    return 'NORMAL'


# ─────────────────────────────────────────────────────────────────────────────
# UNIQUE — batch stock update (LONG — tests slicing)
# No clone. Heavy logging around simple update logic.
# ─────────────────────────────────────────────────────────────────────────────
def process_batch_stock_update(updates, current_inventory):
    logger.info('Starting batch stock update')
    logger.debug(f'Processing {len(updates)} updates')
    logger.debug('Acquiring inventory write lock')
    logger.debug('Lock acquired')
    logger.debug('Loading current inventory snapshot')
    logger.debug('Snapshot loaded')
    logger.info('Validating all update entries before applying')
    logger.debug('Pre-validation pass started')

    success_count = 0
    failure_count = 0
    updated_inventory = dict(current_inventory)

    for update in updates:
        product_id = update['product_id']
        delta = update['delta']
        logger.debug(f'Applying delta {delta} to product {product_id}')
        if product_id not in updated_inventory:
            logger.warning(f'Product {product_id} not found in inventory')
            failure_count += 1
            continue
        new_qty = updated_inventory[product_id] + delta
        if new_qty < 0:
            logger.warning(f'Update would cause negative stock for {product_id}')
            failure_count += 1
            continue
        updated_inventory[product_id] = new_qty
        success_count += 1

    logger.info(f'Batch complete. Success: {success_count} Failures: {failure_count}')
    logger.debug('Releasing inventory write lock')
    logger.debug('Lock released')
    logger.debug('Persisting updated inventory snapshot')
    logger.debug('Inventory snapshot persisted')

    return {
        'updated_inventory': updated_inventory,
        'success_count': success_count,
        'failure_count': failure_count,
    }