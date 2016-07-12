#include <sys/time.h>
#include <time.h>
#include <stdio.h>
static unsigned long long current_time_ns() {
#ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    unsigned long long s = 1000000000ULL * (unsigned long long)mts.tv_sec;
    return (unsigned long long)mts.tv_nsec + s;
#else
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (((unsigned long long)t.tv_nsec)) + s;
#endif
}
/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/**********************************************************************************************/

/* OLDEN parallel C for dynamic structures: compiler, runtime system
 * and benchmarks
 *       
 * Copyright (C) 1994-1996 by Anne Rogers (amr@cs.princeton.edu) and
 * Martin Carlisle (mcc@cs.princeton.edu)
 * ALL RIGHTS RESERVED.
 *
 * OLDEN is distributed under the following conditions:
 *
 * You may make copies of OLDEN for your own use and modify those copies.
 *
 * All copies of OLDEN must retain our names and copyright notice.
 *
 * You may not sell OLDEN or distribute OLDEN in conjunction with a
 * commercial product or service without the expressed written consent of
 * Anne Rogers and Martin Carlisle.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE.
 *
 */


/******************************************************************* 
 *  Health.c : Model of the Colombian Health Care System           *
 *******************************************************************/ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "app-desc.h"
#include "bots.h"
#include "health.h"

/* global variables */
int sim_level;
int sim_cities;
int sim_population_ratio;
int sim_time;
int sim_assess_time;
int sim_convalescence_time;
int32_t sim_seed;
float sim_get_sick_p;
float sim_convalescence_p;
float sim_realloc_p;
int sim_pid = 0;

int res_population;
int res_hospitals;
int res_personnel;
int res_checkin;
int res_village;
int res_waiting;
int res_assess;
int res_inside;
float res_avg_stay;

/**********************************************************
 * Handles math routines for health.c                     *
 **********************************************************/
float my_rand(int32_t *seed) 
{
   int32_t k;
   int32_t idum = *seed;

   idum ^= MASK;
   k = idum / IQ;
   idum = IA * (idum - k * IQ) - IR * k;
   idum ^= MASK;
   if (idum < 0) idum  += IM;
   *seed = idum * IM;
   return (float) AM * idum;
}
/********************************************************************
 * Handles lists.                                                   *
 ********************************************************************/
void addList(struct Patient **list, struct Patient *patient)
{
   if (*list == NULL)
   {
      *list = patient;
      patient->back = NULL;
      patient->forward = NULL;
   }
   else
   {
      struct Patient *aux = *list;
      while (aux->forward != NULL) aux = aux->forward; 
      aux->forward = patient;
      patient->back = aux;
      patient->forward = NULL;
   }
} 
void removeList(struct Patient **list, struct Patient *patient) 
{
#if 0
   struct Patient *aux = *list;
  
   if (patient == NULL) return;
   while((aux != NULL) && (aux != patient)) aux = aux->forward; 

   // Patient not found
   if (aux == NULL) return;

   // Removing patient
   if (aux->back != NULL) aux->back->forward = aux->forward;
   else *list = aux->forward;
   if (aux->forward != NULL) aux->forward->back = aux->back;
#else
   if (patient->back != NULL) patient->back->forward = patient->forward;
   else *list = patient->forward;
   if (patient->forward != NULL) patient->forward->back = patient->back;
#endif
}
/**********************************************************************/
void allocate_village( struct Village **capital, struct Village *back,
   struct Village *next, int level, int32_t vid)
{ 
   int i, population, personnel;
   struct Village *current, *inext;
   struct Patient *patient;

   if (level == 0) *capital = NULL;
   else
   {
      personnel = (int) pow(2, level);
      population = personnel * sim_population_ratio;
      /* Allocate Village */
      *capital = (struct Village *) malloc(sizeof(struct Village));
      /* Initialize Village */
      (*capital)->back  = back;
      (*capital)->next  = next;
      (*capital)->level = level;
      (*capital)->id    = vid;
      (*capital)->seed  = vid * (IQ + sim_seed);
      (*capital)->population = NULL;
      for(i=0;i<population;i++)
      {
         patient = (struct Patient *)malloc(sizeof(struct Patient));
         patient->id = sim_pid++;
         patient->seed = (*capital)->seed;
         // changes seed for capital:
         my_rand(&((*capital)->seed));
         patient->hosps_visited = 0;
         patient->time          = 0;
         patient->time_left     = 0;
         patient->home_village = *capital; 
         addList(&((*capital)->population), patient);
      }
      /* Initialize Hospital */
      (*capital)->hosp.personnel = personnel;
      (*capital)->hosp.free_personnel = personnel;
      (*capital)->hosp.assess = NULL;
      (*capital)->hosp.waiting = NULL;
      (*capital)->hosp.inside = NULL;
      (*capital)->hosp.realloc = NULL;
      const int err = pthread_mutex_init(&(*capital)->hosp.realloc_lock, NULL);
      assert(err == 0);
      // Create Cities (lower level)
      inext = NULL;
      for (i = sim_cities; i>0; i--)
      {
         allocate_village(&current, *capital, inext, level-1, (vid * (int32_t) sim_cities)+ (int32_t) i);
         inext = current;
      }
      (*capital)->forward = current;
   }
}
/**********************************************************************/
struct Results get_results(struct Village *village)
{
   struct Village *vlist;
   struct Patient *p;
   struct Results t_res, p_res;

   t_res.hosps_number     = 0.0;
   t_res.hosps_personnel  = 0.0;
   t_res.total_patients   = 0.0;
   t_res.total_in_village = 0.0;
   t_res.total_waiting    = 0.0;
   t_res.total_assess     = 0.0;
   t_res.total_inside     = 0.0;
   t_res.total_hosps_v    = 0.0;
   t_res.total_time       = 0.0;

   if (village == NULL) return t_res;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
      p_res = get_results(vlist);
      t_res.hosps_number     += p_res.hosps_number;
      t_res.hosps_personnel  += p_res.hosps_personnel;
      t_res.total_patients   += p_res.total_patients;
      t_res.total_in_village += p_res.total_in_village;
      t_res.total_waiting    += p_res.total_waiting;
      t_res.total_assess     += p_res.total_assess;
      t_res.total_inside     += p_res.total_inside;
      t_res.total_hosps_v    += p_res.total_hosps_v;
      t_res.total_time       += p_res.total_time;
      vlist = vlist->next;
   }
   t_res.hosps_number     += 1.0;
   t_res.hosps_personnel  += village->hosp.personnel;

   // Patients in the village
   p = village->population;
   while (p != NULL) 
   {
      t_res.total_patients   += 1.0;
      t_res.total_in_village += 1.0;
      t_res.total_hosps_v    += (float)(p->hosps_visited);
      t_res.total_time       += (float)(p->time); 
      p = p->forward; 
   }
   // Patients in hospital: waiting
   p = village->hosp.waiting;
   while (p != NULL) 
   {
      t_res.total_patients += 1.0;
      t_res.total_waiting  += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time); 
      p = p->forward; 
   }
   // Patients in hospital: assess
   p = village->hosp.assess;
   while (p != NULL) 
   {
      t_res.total_patients += 1.0;
      t_res.total_assess   += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time); 
      p = p->forward; 
   }
   // Patients in hospital: inside
   p = village->hosp.inside;
   while (p != NULL) 
   {
      t_res.total_patients += 1.0;
      t_res.total_inside   += 1.0;
      t_res.total_hosps_v  += (float)(p->hosps_visited);
      t_res.total_time     += (float)(p->time); 
      p = p->forward; 
   }  

   return t_res; 
}
/**********************************************************************/
/**********************************************************************/
/**********************************************************************/
void check_patients_inside(struct Village *village) 
{
   struct Patient *list = village->hosp.inside;
   struct Patient *p;
  
   while (list != NULL)
   {
      p = list;
      list = list->forward; 
      p->time_left--;
      if (p->time_left == 0) 
      {
         village->hosp.free_personnel++;
         removeList(&(village->hosp.inside), p); 
         addList(&(village->population), p); 
      }    
   }
}
/**********************************************************************/
void check_patients_assess_par(struct Village *village) 
{
   struct Patient *list = village->hosp.assess;
   float rand;
   struct Patient *p;

   while (list != NULL) 
   {
      p = list;
      list = list->forward; 
      p->time_left--;

      if (p->time_left == 0) 
      { 
         rand = my_rand(&(p->seed));
         /* sim_covalescense_p % */
         if (rand < sim_convalescence_p)
         {
            rand = my_rand(&(p->seed));
            /* !sim_realloc_p % or root hospital */
            if (rand > sim_realloc_p || village->level == sim_level) 
            {
               removeList(&(village->hosp.assess), p);
               addList(&(village->hosp.inside), p);
               p->time_left = sim_convalescence_time;
               p->time += p->time_left;
            }
            else /* move to upper level hospital !!! */
            {
               village->hosp.free_personnel++;
               removeList(&(village->hosp.assess), p);
               int err = pthread_mutex_lock(&(village->hosp.realloc_lock));
               assert(err == 0);
               addList(&(village->back->hosp.realloc), p); 
               err = pthread_mutex_unlock(&(village->hosp.realloc_lock));
               assert(err == 0);
            } 
         }
         else /* move to village */
         {
            village->hosp.free_personnel++;
            removeList(&(village->hosp.assess), p);
            addList(&(village->population), p); 
         }
      }
   } 
}
/**********************************************************************/
void check_patients_waiting(struct Village *village) 
{
   struct Patient *list = village->hosp.waiting;
   struct Patient *p;
  
   while (list != NULL) 
   {
      p = list;
      list = list->forward; 
      if (village->hosp.free_personnel > 0) 
      {
         village->hosp.free_personnel--;
         p->time_left = sim_assess_time;
         p->time += p->time_left;
         removeList(&(village->hosp.waiting), p);
         addList(&(village->hosp.assess), p); 
      }
      else 
      {
         p->time++;
      }
   } 
}
/**********************************************************************/
void check_patients_realloc(struct Village *village)
{
   struct Patient *p, *s;

   while (village->hosp.realloc != NULL) 
   {
      p = s = village->hosp.realloc;
      while (p != NULL)
      {
         if (p->id < s->id) s = p;
         p = p->forward;
      }
      removeList(&(village->hosp.realloc), s);
      put_in_hosp(&(village->hosp), s);
   }
}
/**********************************************************************/
void check_patients_population(struct Village *village) 
{
   struct Patient *list = village->population;
   struct Patient *p;
   float rand;
  
   while (list != NULL) 
   {
      p = list;
      list = list->forward; 
      /* randomize in patient */
      rand = my_rand(&(p->seed));
      if (rand < sim_get_sick_p) 
      {
         removeList(&(village->population), p);
         put_in_hosp(&(village->hosp), p);
      }
   }

}
/**********************************************************************/
void put_in_hosp(struct Hosp *hosp, struct Patient *patient) 
{  
   (patient->hosps_visited)++;

   if (hosp->free_personnel > 0) 
   {
      hosp->free_personnel--;
      addList(&(hosp->assess), patient); 
      patient->time_left = sim_assess_time;
      patient->time += patient->time_left;
   } 
   else 
   {
      addList(&(hosp->waiting), patient); 
   }
}
/**********************************************************************/
void sim_village_par(struct Village *village)
{
   struct Village *vlist;

   // lowest level returns nothing
   // only for sim_village first call with village = NULL
   // recursive call cannot occurs
   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist)
   {
#pragma omp task untied firstprivate(vlist, village)
sim_village_par(vlist);
      vlist = vlist->next;
   }

   /* Uses lists v->hosp->inside, and v->return */
   check_patients_inside(village);

   /* Uses lists v->hosp->assess, v->hosp->inside, v->population and (v->back->hosp->realloc) !!! */
   check_patients_assess_par(village);

   /* Uses lists v->hosp->waiting, and v->hosp->assess */
   check_patients_waiting(village);

#pragma omp taskwait 
;

   /* Uses lists v->hosp->realloc, v->hosp->asses and v->hosp->waiting */
   check_patients_realloc(village);

   /* Uses list v->population, v->hosp->asses and v->h->waiting */
   check_patients_population(village);
}
/**********************************************************************/
void my_print(struct Village *village)
{
   struct Village *vlist;
   struct Patient *plist;

   if (village == NULL) return;

   /* Traverse village hierarchy (lower level first)*/
   vlist = village->forward;
   while(vlist) {
      my_print(vlist);
      vlist = vlist->next;
   }

   plist = village->population;

   while (plist != NULL) {
      bots_debug("[pid:%d]",plist->id);
      plist = plist->forward; 
   }
   bots_debug("[vid:%d]\n",village->id);

}
/**********************************************************************/
void read_input_data(char *filename)
{
   FILE *fin;
   int res;

   if ((fin = fopen(filename, "r")) == NULL) {
      bots_message("Could not open sequence file (%s)\n", filename);
      exit (-1);
   }
   res = fscanf(fin,"%d %d %d %d %d %d %ld %f %f %f %d %d %d %d %d %d %d %d %f", 
             &sim_level,
             &sim_cities,
             &sim_population_ratio,
             &sim_time, 
             &sim_assess_time,
             &sim_convalescence_time,
             &sim_seed, 
             &sim_get_sick_p,
             &sim_convalescence_p,
             &sim_realloc_p,
             &res_population,
             &res_hospitals,
             &res_personnel,
             &res_checkin,
             &res_village,
             &res_waiting,
             &res_assess,
             &res_inside,
             &res_avg_stay
   );
   if ( res == EOF ) {
      bots_message("Bogus input file (%s)\n", filename);
      exit(-1);
   }
   fclose(fin);

      // Printing input data
   bots_message("\n");
   bots_message("Number of levels    = %d\n", (int) sim_level);
   bots_message("Cities per level    = %d\n", (int) sim_cities);
   bots_message("Population ratio    = %d\n", (int) sim_population_ratio);
   bots_message("Simulation time     = %d\n", (int) sim_time);
   bots_message("Assess time         = %d\n", (int) sim_assess_time);
   bots_message("Convalescence time  = %d\n", (int) sim_convalescence_time);
   bots_message("Initial seed        = %d\n", (int) sim_seed);
   bots_message("Get sick prob.      = %f\n", (float) sim_get_sick_p);
   bots_message("Convalescence prob. = %f\n", (float) sim_convalescence_p);
   bots_message("Realloc prob.       = %f\n", (float) sim_realloc_p);
}
int check_village(struct Village *top)
{
   struct Results result = get_results(top);
   int answer = BOTS_RESULT_SUCCESSFUL;

   if (res_population != result.total_patients) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_hospitals != result.hosps_number) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_personnel != result.hosps_personnel) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_checkin != result.total_hosps_v) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_village != result.total_in_village) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_waiting != result.total_waiting) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_assess != result.total_assess) answer = BOTS_RESULT_UNSUCCESSFUL;
   if (res_inside != result.total_inside) answer = BOTS_RESULT_UNSUCCESSFUL;

   bots_message("\n");
   bots_message("Sim. Variables      = expect / result\n");
   bots_message("Total population    = %6d / %6d people\n", (int)   res_population, (int) result.total_patients);
   bots_message("Hospitals           = %6d / %6d people\n", (int)   res_hospitals, (int) result.hosps_number);
   bots_message("Personnel           = %6d / %6d people\n", (int)   res_personnel, (int) result.hosps_personnel);
   bots_message("Check-in's          = %6d / %6d people\n", (int)   res_checkin, (int) result.total_hosps_v);
   bots_message("In Villages         = %6d / %6d people\n", (int)   res_village, (int) result.total_in_village);
   bots_message("In Waiting List     = %6d / %6d people\n", (int)   res_waiting, (int) result.total_waiting);
   bots_message("In Assess           = %6d / %6d people\n", (int)   res_assess, (int) result.total_assess);
   bots_message("Inside Hospital     = %6d / %6d people\n", (int)   res_inside, (int) result.total_inside);
   bots_message("Average Stay        = %6f / %6f u/time\n", (float) res_avg_stay,(float) result.total_time/result.total_patients);

   my_print(top);

   return answer;
}
/**********************************************************************/
void sim_village_main_par(struct Village *top)
{
    long i;
const unsigned long long full_program_start = current_time_ns();
{
#pragma omp parallel 
{
#pragma omp single 
{
#pragma omp task untied
{
                    for (i = 0; i < sim_time; i++) sim_village_par(top);   
                }
            }
        }
    } ; 
const unsigned long long full_program_end = current_time_ns();
printf("full_program %llu ns", full_program_end - full_program_start);

}

