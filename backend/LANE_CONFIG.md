# Bowling Lane Configuration

## Lane Dimensions

- **Boards per lane**: 39
- **Board width**: ~1.06 inches (total lane width = 41.5 inches)

## Board Numbering

Board numbering depends on bowler handedness:

| Handedness | Board 1 | Board 39 |
|------------|---------|----------|
| Right-handed | Far right (right gutter edge) | Far left (left gutter edge) |
| Left-handed | Far left (left gutter edge) | Far right (right gutter edge) |

From the **behind-the-bowler camera view**:
- Right-handed: board 1 is on the **right** side of the image
- Left-handed: board 1 is on the **left** side of the image

## Approach Dots

Bowling lanes have rows of alignment dots on the approach (before the foul line).
These dots are used by bowlers to line up their starting position and target.

### Dot Configurations

| Config | Dot Count | Board Numbers | Notes |
|--------|-----------|---------------|-------|
| 5-dot  | 5 | 10, 15, 20, 25, 30 | Common in many houses |
| 7-dot  | 7 | 5, 10, 15, 20, 25, 30, 35 | Includes extra outer dots |

- All dots are spaced **5 boards apart**
- The **center dot** is always at **board 20**
- Dot rows are typically at 12 ft and 15 ft from the foul line

### Current Configuration

**This bowling alley uses 5-dot rows.**

## Configuration File

Lane-specific settings are stored in `lane_config.json`.
Detection code reads from this file at startup.
