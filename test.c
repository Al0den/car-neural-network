#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

struct Graphe {
    int V;
    int E;
    int *adj;
};

struct Arete {
    int s1;
    int s2;
    int p;
};

// Question 15

struct Arete* liste_aretes(struct Graphe g) {
    struct Arete* liste = malloc(sizeof(struct Arete) * (int)(g.V * (g.V - 1)/2));
    assert(liste != NULL);
    int index = 0;
    for (int i=0; i<g.V; i++) {
        for (int j=0; j<g.V; i++) {
            if (g.adj[i * g.V + j] != 0) {
                struct Arete new_arr = {.s1 = i, .s2 = j, .p = g.adj[i * g.V + j]};
                liste[index++] = new_arr;
            }
        }
    }
    return liste;
}

// Question 16

struct Graphe* init_kruskal(struct Graphe g) {
    struct Graphe* res = malloc(sizeof(struct Graphe));
    assert(res != NULL);
    res->V = g.V;
    res->E = g.V - 1;
    res->adj = malloc(sizeof(int*) * g.V * g.V);
    assert(res->adj != NULL);
    for (int i=0; i<g.V; i++) {
        for (int j=0; j<g.V; j++) {
            res->adj[i * g.V + j] = 0;
        }
    }
    return res;
}

void tri_aretes(struct Arete a[], int k);

struct Graphe kruskal(struct Graphe g) {
    struct Graphe* res = init_kruskal(g);
    struct Arete* aretes = liste_aretes(g);
    int nb_aretes = g.V * (g.V - 1) / 2;
    tri_aretes(aretes, nb_aretes);
    int* parent = malloc(sizeof(int) * g.V);
    assert(parent != NULL);
    for (int i = 0; i < g.V; i++) {
        parent[i] = i;
    }
    int index = 0; 
    for (int i = 0; i < nb_aretes; i++) {
        int u = aretes[i].s1;
        int v = aretes[i].s2;
        int u_ens = find(parent, u);
        int v_ens = find(parent, v);
        if (u_ens != v_ens) {
            res->adj[u * g.V + v] = 1;
            res->adj[v * g.V + u] = 1;
            // res->adj[u * g.V + v] = aretes[i].p;
            // res->adj[v * g.V + u] = aretes[i].p;
            union_sets(parent, u_ens, v_ens);
            index++;
        }
        if (index == g.V - 1) {
            break;
        }
    }
    free(parent);
    free(aretes);
    return *res;
}

int find(int parent[], int i) {
    if (parent[i] == i) {
        return i;
    }
    return find(parent, parent[i]);
}

void union_sets(int parent[], int u, int v) {
    int u_set = find(parent, u);
    int v_set = find(parent, v);
    parent[u_set] = v_set;
}

// Question 18

int degre(struct Graphe g, int i) {
    int s = 0;
    for (int j =0; j<g.V - 1; j++) {
        if (g.adj[i * g.V + j] != 0) {
            s = s + 1;
        }
    }
    return s;
}

// Question 19

int *sommets_impairs(struct Graphe g, int *nb_sommets) {
    int *degres = malloc(sizeof(int) * g.V);
    assert(degres != NULL);

    int degres_impairs = 0;
    for (int i=0; i<g.V; i++) {
        int deg = degre(g, i);
        degres[i] = deg;
        if (deg % 2 == 0) {
            degres_impairs++;
        }
    }
    *nb_sommets = degres_impairs;
    int *res = malloc(sizeof(int) * degres_impairs);
    int ind = 0;
    for (int i=0; i<g.V; i++) {
        if (degres[i] % 2 == 0) {
            res[ind++] = i;
        }
    }
    return res;
}

int main() {}




