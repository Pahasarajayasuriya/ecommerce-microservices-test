// user-service/userService.js
// Handles user profile management, search, pagination

const logger = require('./logger');


// ─────────────────────────────────────────────────────────────────────────────
// CLONE GROUP D — "paginated list builder"
// Semantic clone of: payment-service/checkoutService.py::get_transaction_page()
// Same logic: offset/limit slice, total_pages computation
// Different domain: user list vs transaction list
// ─────────────────────────────────────────────────────────────────────────────
function paginateResults(items, page, pageSize) {
    const totalCount = items.length;
    const offset = (page - 1) * pageSize;
    const pageItems = items.slice(offset, offset + pageSize);
    const totalPages = Math.ceil(totalCount / pageSize);
    return {
        items: pageItems,
        page,
        pageSize,
        totalCount,
        totalPages,
    };
}


// ─────────────────────────────────────────────────────────────────────────────
// UNIQUE — user profile merge
// No clone. User-service-specific field merge logic.
// ─────────────────────────────────────────────────────────────────────────────
function mergeProfileUpdates(existingProfile, updates) {
    const merged = Object.assign({}, existingProfile);
    for (const key of Object.keys(updates)) {
        if (updates[key] !== null && updates[key] !== undefined) {
            merged[key] = updates[key];
        }
    }
    merged.updatedAt = new Date().toISOString();
    return merged;
}


// ─────────────────────────────────────────────────────────────────────────────
// UNIQUE — search filter builder (LONG — tests slicing)
// No clone. Complex filter construction unique to user search.
// ─────────────────────────────────────────────────────────────────────────────
function buildUserSearchQuery(filters) {
    logger.info('Building user search query');
    logger.debug('Input filters: ' + JSON.stringify(filters));
    logger.debug('Initialising query builder');
    logger.debug('Setting default sort order');
    logger.debug('Applying pagination defaults');
    logger.debug('Checking filter schema');
    logger.debug('Filter schema validated');
    logger.info('Processing individual filter fields');

    const query = {};
    let filterCount = 0;

    if (filters.name && filters.name.length > 0) {
        query.name = { $regex: filters.name, $options: 'i' };
        filterCount += 1;
        logger.debug('Name filter applied');
    }
    if (filters.email && filters.email.length > 0) {
        query.email = filters.email.toLowerCase();
        filterCount += 1;
        logger.debug('Email filter applied');
    }
    if (filters.role && filters.role.length > 0) {
        query.role = filters.role;
        filterCount += 1;
        logger.debug('Role filter applied');
    }
    if (filters.createdAfter) {
        query.createdAt = { $gte: new Date(filters.createdAfter) };
        filterCount += 1;
        logger.debug('Date filter applied');
    }

    logger.info('Query built with ' + filterCount + ' active filters');
    logger.debug('Serialising query object');
    logger.debug('Query serialised successfully');
    logger.debug('Returning final query to caller');

    return query;
}


module.exports = { paginateResults, mergeProfileUpdates, buildUserSearchQuery };