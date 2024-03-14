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

struct Graphe kruskal(struct Graphe g) {
    struct Arete* liste = liste_aretes(g);
    struct Graphe* res = init_kruskal(g);
    int* comp = malloc(sizeof(int) * g.V);
    assert(comp != NULL);
    for (int i=0; i<g.V; i++) {
        comp[i] = i;
    }
    for (int i=0; i<g.V; i++) {
        int s1 = liste[i].s1;
        int s2 = liste[i].s2;
        if (comp[s1] != comp[s2]) {
            res->adj[s1 * g.V + s2] = liste[i].p;
            res->adj[s2 * g.V + s1] = liste[i].p;
            int old_comp = comp[s2];
            for (int j=0; j<g.V; j++) {
                if (comp[j] == old_comp) {
                    comp[j] = comp[s1];
                }
            }
        }
    }
    free(comp);
    free(liste);
    return *res;
}

int degre(struct Graphe g, int i) {
    int s = 0;
    for (int j =0; j<g.V - 1; j++) {
        if (g.adj[i * g.V + j] != 0) {
            s = s + 1;
        }
    }
    return s;
}

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




