__kernel void dt_mnist(__global const float *x, __global int *restrict z)
{
	if (x[350] <= 131.5) {
		if (x[568] <= 0.5) {
			if (x[430] <= 0.5) {
				if (x[405] <= 2.5) {
					z[0] = 7;
				}
				else {
					z[0] = 8;
				}
			}
			else {
				if (x[211] <= 28.5) {
					z[0] = 4;
				}
				else {
					z[0] = 9;
				}
			}
		}
		else {
			if (x[435] <= 0.5) {
				if (x[489] <= 22.5) {
					z[0] = 0;
				}
				else {
					z[0] = 2;
				}
			}
			else {
				if (x[346] <= 0.5) {
					z[0] = 2;
				}
				else {
					z[0] = 6;
				}
			}
		}
	}
	else {
		if (x[489] <= 26.5) {
			if (x[290] <= 34.5) {
				if (x[486] <= 58.5) {
					z[0] = 3;
				}
				else {
					z[0] = 8;
				}
			}
			else {
				if (x[297] <= 5.5) {
					z[0] = 5;
				}
				else {
					z[0] = 9;
				}
			}
		}
		else {
			if (x[234] <= 0.5) {
				if (x[402] <= 0.5) {
					z[0] = 1;
				}
				else {
					z[0] = 6;
				}
			}
			else {
				if (x[658] <= 0.5) {
					z[0] = 2;
				}
				else {
					z[0] = 8;
				}
			}
		}
	}
}
