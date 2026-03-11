// user-service/loyaltyService.java
// Handles loyalty points, member discounts, tier management

import java.util.logging.Logger;

public class loyaltyService {

    private static final Logger logger =
        Logger.getLogger(loyaltyService.class.getName());

    // ─────────────────────────────────────────────────────────────────────────
    // CLONE GROUP B — "percentage discount"
    // Semantic clone of: order-service/orderService.js::applyDiscountCode()
    // Same logic: compute discount amount, subtract from original
    // Different language: Java vs JS, different variable names
    // ─────────────────────────────────────────────────────────────────────────
    public double applyMemberDiscount(double basePrice, double memberDiscountPct) {
        double savings = basePrice * (memberDiscountPct / 100.0);
        double discountedPrice = basePrice - savings;
        return discountedPrice;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UNIQUE — points accrual calculation
    // No clone. Loyalty-specific tier multiplier logic.
    // ─────────────────────────────────────────────────────────────────────────
    public int calculatePointsEarned(double purchaseAmount,
                                      String memberTier) {
        double baseRate = 1.0;
        switch (memberTier) {
            case "SILVER": baseRate = 1.5; break;
            case "GOLD":   baseRate = 2.0; break;
            case "PLATINUM": baseRate = 3.0; break;
        }
        int points = (int) Math.floor(purchaseAmount * baseRate);
        logger.info("Points earned: " + points + " for tier " + memberTier);
        return points;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UNIQUE — tier upgrade eligibility
    // No clone. Domain-specific threshold logic.
    // ─────────────────────────────────────────────────────────────────────────
    public String evaluateTierUpgrade(int currentPoints, String currentTier) {
        if (currentTier.equals("BRONZE") && currentPoints >= 1000) {
            return "SILVER";
        }
        if (currentTier.equals("SILVER") && currentPoints >= 5000) {
            return "GOLD";
        }
        if (currentTier.equals("GOLD") && currentPoints >= 20000) {
            return "PLATINUM";
        }
        return currentTier;
    }
}