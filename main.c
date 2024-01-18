#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>

#define NB_ETUDIANTS 50
#define NB_LAVEURS 10

struct Pile {
    int assiette;
    struct Pile * suite;
};

typedef struct Pile pile;

int prendre(pile **p, pthread_mutex_t *m) {
    pthread_mutex_lock(m);
    if (*p == NULL) {
        pthread_mutex_unlock(m);
        return -1;
    }
    pile * tmp = *p;
    *p = (*p)->suite;
    pthread_mutex_unlock(m);
    int assiette = tmp->assiette;
    return assiette;
}

void poser(pile **p, pthread_mutex_t *m, int assiette) {
    pthread_mutex_lock(m);
    pile * tmp = (pile *) malloc(sizeof(pile));
    tmp->assiette = assiette;
    tmp->suite = *p;
    *p = tmp;
    pthread_mutex_unlock(m);
}

typedef struct args {
    pile * p_propre;
    pile * p_sale;
    pthread_mutex_t * m_propre;
    pthread_mutex_t * m_sale;
    int thread_id;
} arg;

void * manger(void *args) {
    while (true) {
        arg * a = (arg *) args;
        int assiette = prendre(&a->p_propre, a->m_propre);
        if (assiette == -1) {
            printf("L'etudiant %d n'a pas pu prendre d'assiette propre\n", a->thread_id);
        }
        printf("L'etudiant %d a fini de manger\n", a->thread_id);
        poser(&a->p_sale, a->m_sale, assiette);
    }
    return NULL;
}

void * laver(void *args) {
    while(true) {
        arg * a = (arg *) args;
        int assiette = prendre(&a->p_sale, a->m_sale);
        if (assiette == -1) {
            printf("Le laveur %d n'a pas pu prendre d'assiette sale\n", a->thread_id);
        }
        printf("Le laveur %d a fini de laver\n", a->thread_id);
        poser(&a->p_propre, a->m_propre, assiette);
    }
    return NULL;
}

int main() {
    pthread_mutex_t m_propre = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t m_sale = PTHREAD_MUTEX_INITIALIZER;
    pile * p_propre = (pile *) malloc(sizeof(pile));
    pile * p_sale = (pile *) malloc(sizeof(pile));
    p_propre->assiette = 0;
    p_propre->suite = NULL;

    for (int i = 1; i < 10; ++i) {
        poser(&p_propre, &m_propre, i);
    }

    pthread_t etudiants[NB_ETUDIANTS];
    pthread_t laveurs[NB_LAVEURS];
    
    for (int i = 0; i < NB_ETUDIANTS; ++i) {
        arg * a = (arg *) malloc(sizeof(arg));
        a->p_propre = p_propre;
        a->p_sale = p_sale;
        a->m_propre = &m_propre;
        a->m_sale = &m_sale;
        a->thread_id = i;
        pthread_create(&etudiants[i], NULL, manger, a);
    }

    for (int i = 0; i < NB_LAVEURS; ++i) {
        arg * a = (arg *) malloc(sizeof(arg));
        a->p_propre = p_propre;
        a->p_sale = p_sale;
        a->m_propre = &m_propre;
        a->m_sale = &m_sale;
        a->thread_id = i;
        pthread_create(&laveurs[i], NULL, laver, a);
    }

    for (int i = 0; i < NB_ETUDIANTS; ++i) {
        pthread_join(etudiants[i], NULL);
        if(i<10) {
            pthread_join(laveurs[i], NULL);
        }
    }
    
    return 0;
}

