(function() {
  'use strict';

  angular
    .module('bubbleApp')
    .config(bubbleAppRoutes);

  bubbleAppRoutes.$inject = [
    '$stateProvider',
    '$urlRouterProvider'
  ];

  function bubbleAppRoutes($stateProvider, $urlRouterProvider) {
    $urlRouterProvider.otherwise('/home');

    $stateProvider
      .state('home', {
      url: '/home',
      controller: 'TweetListCtrl as vm',
      templateUrl: 'app/tweetlist/tweetlist.html'
    });
  }
})();