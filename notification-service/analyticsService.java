// notification-service/analyticsService.java
// Handles notification delivery, analytics scoring, alert routing

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class analyticsService {

    private static final Logger logger =
        Logger.getLogger(analyticsService.class.getName());

    // ─────────────────────────────────────────────────────────────────────────
    // CLONE GROUP E — "weighted average calculation"
    // Semantic clone of: inventory-service/stockService.py::calculate_weighted_average_cost()
    // Same logic: multiply values by weights, sum, divide by total weight
    // Different domain: notification urgency score vs stock cost
    // ─────────────────────────────────────────────────────────────────────────
    public double computeWeightedScore(List<Map<String, Double>> signals) {
        double totalScore = 0;
        double totalWeight = 0;
        for (Map<String, Double> signal : signals) {
            totalScore += signal.get("value") * signal.get("weight");
            totalWeight += signal.get("weight");
        }
        if (totalWeight == 0) {
            return 0;
        }
        double weightedScore = totalScore / totalWeight;
        return weightedScore;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UNIQUE — notification routing by channel priority
    // No clone. Business logic specific to notification delivery rules.
    // ─────────────────────────────────────────────────────────────────────────
    public String resolveNotificationChannel(String eventType,
                                              String userPreference,
                                              boolean isUrgent) {
        if (isUrgent) {
            return "SMS";
        }
        if (userPreference != null && !userPreference.isEmpty()) {
            return userPreference;
        }
        switch (eventType) {
            case "ORDER_SHIPPED":    return "EMAIL";
            case "PAYMENT_FAILED":  return "SMS";
            case "PROMO_OFFER":     return "PUSH";
            default:                return "EMAIL";
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CLONE GROUP F — "retry with exponential backoff"
    // Semantic clone of: user-service/authService.java::retryWithBackoff()
    // Same logic: loop up to maxAttempts, double delay each iteration
    // Different domain: notification delivery vs auth token refresh
    // ─────────────────────────────────────────────────────────────────────────
    public boolean deliverWithRetry(String notificationId,
                                     String payload,
                                     int maxAttempts) {
        int attempt = 0;
        int delayMs = 100;
        while (attempt < maxAttempts) {
            attempt += 1;
            logger.info("Delivery attempt " + attempt +
                        " for notification " + notificationId);
            boolean success = sendToChannel(payload);
            if (success) {
                logger.info("Delivered successfully on attempt " + attempt);
                return true;
            }
            logger.warning("Attempt " + attempt + " failed. " +
                           "Retrying in " + delayMs + "ms");
            delayMs = delayMs * 2;
        }
        logger.severe("All " + maxAttempts +
                      " delivery attempts failed for " + notificationId);
        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UNIQUE — alert suppression window check
    // No clone. Notification-specific deduplication logic.
    // ─────────────────────────────────────────────────────────────────────────
    public boolean isWithinSuppressionWindow(long lastSentTimestamp,
                                              long currentTimestamp,
                                              long windowMs) {
        long elapsed = currentTimestamp - lastSentTimestamp;
        return elapsed < windowMs;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UNIQUE — batch notification dispatch (LONG — tests slicing)
    // No clone. Heavy orchestration with logging.
    // ─────────────────────────────────────────────────────────────────────────
    public Map<String, Integer> dispatchBatchNotifications(
            List<Map<String, String>> notifications,
            Map<String, Boolean> userOptOuts) {

        logger.info("Starting batch notification dispatch");
        logger.fine("Total notifications to process: " + notifications.size());
        logger.fine("Loading user opt-out preferences");
        logger.fine("Opt-out map loaded with " + userOptOuts.size() + " entries");
        logger.info("Beginning dispatch loop");
        logger.fine("Checking suppression windows before dispatch");

        int sent = 0;
        int skipped = 0;
        int failed = 0;

        for (Map<String, String> notification : notifications) {
            String userId = notification.get("user_id");
            String channel = notification.get("channel");
            logger.fine("Processing notification for user: " + userId);

            if (userOptOuts.getOrDefault(userId, false)) {
                logger.fine("User " + userId + " opted out. Skipping.");
                skipped += 1;
                continue;
            }

            boolean delivered = deliverWithRetry(
                notification.get("id"), notification.get("payload"), 3);

            if (delivered) {
                sent += 1;
            } else {
                logger.warning("Failed to deliver to user: " + userId);
                failed += 1;
            }
        }

        logger.info("Batch dispatch complete. " +
                    "Sent=" + sent + " Skipped=" + skipped +
                    " Failed=" + failed);
        logger.fine("Writing dispatch summary to audit log");
        logger.fine("Audit log entry written");
        logger.fine("Releasing dispatch resources");

        return Map.of("sent", sent, "skipped", skipped, "failed", failed);
    }

    // Stub — represents actual channel delivery (external call)
    private boolean sendToChannel(String payload) {
        return true;
    }
}