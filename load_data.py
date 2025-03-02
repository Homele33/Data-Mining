import csv


def load_points(in_path, dim, n=-1, points=[]):
    """
    Load points from a CSV file at in_path, extracting up to n points with dim dimensions.

    Parameters:
        in_path (str): Path to the CSV file.
        dim (int): Number of dimensions to extract from each row.
        n (int, optional): Number of points to load (-1 loads all). Defaults to -1.
        points (list, optional): List to store the loaded points. Defaults to an empty list.

    Returns:
        list: The updated list containing the loaded points.
    """
    with open(in_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        count = 0
        for row in reader:
            if 0 <= n == count:
                break
            points.append(tuple([float(value) for value in row[:dim]]))
            count += 1
    return points
