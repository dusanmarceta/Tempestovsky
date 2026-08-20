import numpy as np
import matplotlib.pyplot as plt



def load_multiple_npy(filename):
    """Load all NumPy arrays stored sequentially in one file."""
    arrays = []

    with open(filename, "rb") as f:
        while True:
            try:
                arrays.append(np.load(f))
            except (EOFError, ValueError):
                break

    return arrays


def load_version(version):
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    insolation_sections = load_multiple_npy(
        f"output/precomputed_insolation_{version}.npy"
    )

    true_anomaly_sections = load_multiple_npy(
        f"output/true_anomaly_{version}.npy"
    )

    sun_distance_sections = load_multiple_npy(
        f"output/current_sun_distance_{version}.npy"
    )

    r_rad_sections = load_multiple_npy(
        f"output/r_rad_{version}.npy"
    )

    r_trans_sections = load_multiple_npy(
        f"output/r_trans_{version}.npy"
    )

    last_state_sections = load_multiple_npy(
        f"output/last_state_{version}.npy"
    )

    # --------------------------------------------------
    # Check what was loaded
    # --------------------------------------------------

    print(f"\nVersion: {version}")
    print("Number of orbit sections:")
    print("  insolation:", len(insolation_sections))
    print("  true anomaly:", len(true_anomaly_sections))
    print("  sun distance:", len(sun_distance_sections))
    print("  r_rad:", len(r_rad_sections))
    print("  r_trans:", len(r_trans_sections))
    print("  last state:", len(last_state_sections))

    # --------------------------------------------------
    # Combine orbit sections
    # --------------------------------------------------

    precomputed_insolation = np.concatenate(
        insolation_sections,
        axis=1
    )

    true_anomaly = np.concatenate(
        true_anomaly_sections
    )

    current_sun_distance = np.concatenate(
        sun_distance_sections
    )

    r_rad = np.concatenate(
        r_rad_sections,
        axis=0
    )

    r_trans = np.concatenate(
        r_trans_sections,
        axis=0
    )

    # --------------------------------------------------
    # Last state
    # --------------------------------------------------

    last_state = last_state_sections[-1]

    # --------------------------------------------------
    # Final shapes
    # --------------------------------------------------

    print("\nFinal arrays:")
    print("precomputed_insolation:", precomputed_insolation.shape)
    print("true_anomaly:", true_anomaly.shape)
    print("current_sun_distance:", current_sun_distance.shape)
    print("r_rad:", r_rad.shape)
    print("r_trans:", r_trans.shape)
    print("last_state:", last_state.shape)

    return (
        precomputed_insolation,
        true_anomaly,
        current_sun_distance,
        r_rad,
        r_trans,
        last_state
    )


# ==================================================
# OLD
# ==================================================

(
    precomputed_insolation_old,
    true_anomaly_old,
    current_sun_distance_old,
    r_rad_old,
    r_trans_old,
    last_state_old
) = load_version("old")


# ==================================================
# NEW
# ==================================================

(
    precomputed_insolation_new,
    true_anomaly_new,
    current_sun_distance_new,
    r_rad_new,
    r_trans_new,
    last_state_new
) = load_version("new")


# ==================================================
# Coordinates
# ==================================================

long_r_old = np.rad2deg(
    np.arctan2(r_rad_old[:, 1], r_rad_old[:, 0])
)

lat_r_old = np.rad2deg(
    np.arcsin(r_rad_old[:, 2])
)

long_t_old = np.rad2deg(
    np.arctan2(r_trans_old[:, 1], r_trans_old[:, 0])
)

lat_t_old = np.rad2deg(
    np.arcsin(r_trans_old[:, 2])
)


long_r_new = np.rad2deg(
    np.arctan2(r_rad_new[:, 1], r_rad_new[:, 0])
)

lat_r_new = np.rad2deg(
    np.arcsin(r_rad_new[:, 2])
)

long_t_new = np.rad2deg(
    np.arctan2(r_trans_new[:, 1], r_trans_new[:, 0])
)

lat_t_new = np.rad2deg(
    np.arcsin(r_trans_new[:, 2])
)


mean_insol_old = precomputed_insolation_old.mean(axis=1)
mean_insol_new = precomputed_insolation_new.mean(axis=1)

step = 1479


plt.figure()
plt.title('x RAD (temepstovsky)', fontsize = 24)
plt.plot(r_rad_old[:, 0], label = 'old')
plt.plot(r_rad_new[:, 0], label = 'new')
plt.legend()
plt.grid()


plt.figure()
plt.title('y RAD (temepstovsky)', fontsize = 24)
plt.plot(r_rad_old[:, 1], label = 'old')
plt.plot(r_rad_new[:, 1], label = 'new')
plt.legend()
plt.grid()

plt.figure()
plt.title('z RAD (temepstovsky)', fontsize = 24)
plt.plot(r_rad_old[:, 2], label = 'old')
plt.plot(r_rad_new[:, 2], label = 'new')
plt.legend()
plt.grid()



plt.figure()
plt.title('x TRANS (temepstovsky)', fontsize = 24)
plt.plot(r_trans_old[:, 0], label = 'old')
plt.plot(r_trans_new[:, 0], label = 'new')
plt.legend()
plt.grid()


plt.figure()
plt.title('y TRANS (temepstovsky)', fontsize = 24)
plt.plot(r_trans_old[:, 1], label = 'old')
plt.plot(r_trans_new[:, 1],  label = 'new')
plt.legend()
plt.grid()

plt.figure()
plt.title('z TRANS (temepstovsky)', fontsize = 24)
plt.plot(r_trans_old[:, 2], label = 'old')
plt.plot(r_trans_new[:, 2], 'o', label='new')

idx_red = np.arange(0, len(r_trans_new), step)
idx_square = np.arange(step - 1, len(r_trans_new), step)

plt.plot(idx_red, r_trans_new[idx_red, 2], 'ro')
plt.plot(idx_square, r_trans_new[idx_square, 2], 's')
plt.legend()
plt.grid()

# plt.figure()
# plt.title('long (temepstovsky)', fontsize = 24)
# plt.plot(long_r_old)

# plt.figure()
# plt.title('lat (temepstovsky)', fontsize = 24)
# plt.plot(lat_r_old)