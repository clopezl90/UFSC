
def move_particles(positions, VelocityField, delta_t):
    new_positions = []
    for x_old,y_old in positions:
        u=VelocityField.u_field[int(x_old), int(y_old)]
        v=VelocityField.v_field[int(x_old), int(y_old)]

        x_new= x_old + u*delta_t
        y_new= y_old + v*delta_t

        new_positions.append([x_new, y_new])
    return new_positions
    #return np.array(new_positions, dtype=int) esta opcion es de SOverflow.


