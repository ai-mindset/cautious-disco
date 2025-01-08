//I've generated a synthetic time series with 100 data points that incorporates several key patterns observed in my real-world examples:
//
// A general upward trend
// A significant step change around day 30 (similar to what we see in real data when there are structural breaks)
// Another step change around day 60
// A cyclical component (similar to weekly/monthly patterns in real data)
// Small random variations to simulate noise
//
// The data has two columns:
//
// date: in DD/MM/YYYY format
// dailyTotal: numeric dailyTotals starting around 100 and ending around 180
//
// Key characteristics:
//
// Starting dailyTotal: ~100
// Ending dailyTotal: ~180
// Major step changes: +20 points and +15 points at two different points
// Natural variation: ±1.5 points
// Underlying trend: +0.5 points per day
// Cyclical component: amplitude of ~10 points
//
// This synthetic series captures the essence of real-world patterns while being distinct from my source data.

// Generate synthetic data with similar patterns
const startDate = new Date("2023-01-01");
const data = [];

for (let i = 0; i < 100; i++) {
  const currentDate = new Date(startDate);
  currentDate.setDate(startDate.getDate() + i);

  // Base trend
  let dailyTotal = 100 + (i * 0.5);

  // Add cyclical component
  dailyTotal += 10 * Math.sin(i * 0.1);

  // Add step changes at specific points
  if (i > 30) dailyTotal += 20;
  if (i > 60) dailyTotal += 15;

  // Add small random noise
  dailyTotal += (Math.random() - 0.5) * 3;

  // Format date as DD/MM/YYYY
  const dateStr = currentDate.toLocaleDateString("en-GB");

  data.push(`${dateStr},${dailyTotal.toFixed(1)}`);
}

// Create CSV content with header
const csvContent = "date,daily_total\n" + data.join("\n");
console.log(csvContent);
